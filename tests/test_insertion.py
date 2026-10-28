"""The insertion module: the scanners, the editor and the verifier that proves an edit changes nothing else.

Fuzz tests use seeded generators, so a failure reproduces. No network.
"""

import ast
import copy
import json
import random
import unittest

from flipjev import insertion as IP
from tests.fixtures import trajectory

PIECES = ["a", "b", "word", " ", "  ", "\n", "\t", "'", '"', "\\", "\\\\", "{", "}", "[", "]", ",", ":", "é",
          "\U0001f600", " ", "\x00", "\x01", "%s", "{x}", "\\n", "###", "...", "'''", '"""', "</s>", "\\'", '\\"']


def rand_text(rng, limit=10):
    return "".join(rng.choice(PIECES) for _ in range(rng.randint(0, limit)))


def rand_obj(rng, depth=0):
    kinds = ["str", "str", "str", "int", "float", "bool", "none"] + (["dict", "list"] * 2 if depth < 4 else [])
    kind = rng.choice(kinds)
    if kind == "str":
        return rand_text(rng)
    if kind == "int":
        return rng.randint(-5, 10 ** 6)
    if kind == "float":
        return rng.choice([0.5, -2.25, 1e-7, 3.0])
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "none":
        return None
    if kind == "dict":
        return {rand_text(rng, 4) + str(i): rand_obj(rng, depth + 1) for i in range(rng.randint(0, 4))}
    return [rand_obj(rng, depth + 1) for _ in range(rng.randint(0, 4))]


def reference_leaves(obj, path=()):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from reference_leaves(value, path + (key,))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from reference_leaves(value, path + (index,))


def set_path(obj, path, value):
    if not path:
        return value
    node = obj
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    return obj


class ScannerTests(unittest.TestCase):
    def test_json_scanner_agrees_with_the_standard_library(self):
        rng = random.Random(7)
        for _ in range(200):
            text = json.dumps(rand_obj(rng), ensure_ascii=rng.random() < 0.5, indent=rng.choice([None, 2]))
            leaves = IP.json_leaves(text)
            self.assertEqual([(l.path, l.value) for l in leaves], list(reference_leaves(json.loads(text))))
            for leaf in leaves:
                self.assertEqual(json.loads('"' + text[leaf.start:leaf.end] + '"'), leaf.value)

    def test_json_scanner_rejects_what_json_rejects(self):
        for bad in ["{'a': 1}", '{"a": 1,}', "[1 2]", '{"a": 1} junk', ""]:
            with self.assertRaises(ValueError):
                IP.json_leaves(bad)

    def test_python_literal_scanner_agrees_with_literal_eval(self):
        rng = random.Random(9)
        for _ in range(200):
            text = repr(rand_obj(rng))
            if "\r" in text or IP.detect_format(text) != "pylit":
                continue
            leaves = IP.pylit_leaves(text)
            self.assertEqual([(l.path, l.value) for l in leaves], list(reference_leaves(ast.literal_eval(text))))

    def test_reprlike_values_are_closed_by_the_following_key_or_bracket(self):
        text = "{'body': 'Please read the 'report' now.', 'status': 'ok'}"
        leaves = IP.reprlike_leaves(text)
        self.assertEqual(leaves[0].value, "Please read the 'report' now.")
        self.assertEqual(leaves[1].value, "ok")

    def test_detect_format(self):
        self.assertEqual(IP.detect_format('{"a": "b"}'), "json")
        self.assertEqual(IP.detect_format("{'a': 'b'}"), "pylit")
        self.assertEqual(IP.detect_format("{'a': 'it's here', 'b': 'x'}"), "reprlike")
        self.assertEqual(IP.detect_format("just prose here"), "text")
        self.assertEqual(IP.detect_format("   "), "empty")


class EditTests(unittest.TestCase):
    def oracle(self, text, fmt, parse, suffix):
        old_obj = parse(text)
        for leaf in IP.leaves_of(text, fmt):
            new_text, inserted, got = IP.edit_container(text, fmt, leaf.path, suffix)
            sep = IP._separator(leaf.value)
            self.assertEqual(new_text[:got.end], text[:got.end])
            self.assertEqual(new_text[got.end + len(inserted):], text[got.end:])
            expected = set_path(copy.deepcopy(old_obj), leaf.path, leaf.value + sep + suffix)
            self.assertEqual(repr(parse(new_text)), repr(expected), (leaf.path, suffix))
            self.assertEqual(IP.detect_format(new_text), fmt)

    def test_json_fuzz(self):
        rng = random.Random(11)
        for _ in range(120):
            text = json.dumps(rand_obj(rng), ensure_ascii=rng.random() < 0.5, indent=rng.choice([None, 2]))
            for suffix in (IP.PROBE, rand_text(rng, 14) or "x", "plain words"):
                self.oracle(text, "json", json.loads, suffix)

    def test_pylit_fuzz(self):
        rng = random.Random(13)
        for _ in range(120):
            text = repr(rand_obj(rng))
            if "\r" in text or IP.detect_format(text) != "pylit":
                continue
            for suffix in (IP.PROBE, rand_text(rng, 14) or "x", "plain words"):
                self.oracle(text, "pylit", lambda t: ast.literal_eval(t.strip()), suffix)

    def test_text_container_is_a_plain_append_with_a_separator(self):
        self.assertEqual(IP.edit_container("Result: ok.", "text", [], "note: x")[0], "Result: ok. note: x")
        self.assertEqual(IP.edit_container("line one\nline two", "text", [], "note: x")[0], "line one\nline two\n\nnote: x")
        self.assertEqual(IP.edit_container("", "text", [], "note: x")[0], "note: x")

    def test_json_edit_lands_inside_the_string(self):
        new, _, _ = IP.edit_container('{"body": "Hello there.", "n": 3}', "json", ["body"], 'say "hi"\n\\')
        self.assertEqual(json.loads(new)["body"], 'Hello there. say "hi"\n\\')

    def test_a_stale_slot_is_a_slot_error(self):
        with self.assertRaises(IP.SlotError):
            IP.edit_container("not json", "json", ["a"], "s")


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.traj = trajectory()
        slots, _ = IP.enumerate_slots(self.traj)
        self.slot = next(s for s in slots if s["index"] == 2 and s["path"] == ["email", "body"])

    def test_a_clean_edit_has_no_problems(self):
        edited = IP.apply_slot(self.traj, self.slot, "extra words here", verify=False)
        self.assertEqual(IP.verify_edit(self.traj, edited, self.slot, "extra words here"), [])

    def test_naive_append_after_the_document_is_caught(self):
        content = self.traj["messages"][2]["content"]
        naive = {**self.traj, "messages": list(self.traj["messages"])}
        naive["messages"][2] = {**naive["messages"][2], "content": content + " extra words here"}
        problems = IP.verify_edit(self.traj, naive, self.slot, "extra words here")
        self.assertTrue(any("format changed" in p for p in problems), problems)

    def test_editing_a_different_string_is_caught(self):
        slots, _ = IP.enumerate_slots(self.traj)
        other = next(s for s in slots if s["index"] == 2 and s["path"] == ["email", "subject"])
        wrong = IP.apply_slot(self.traj, other, "extra words here")
        self.assertTrue(IP.verify_edit(self.traj, wrong, self.slot, "extra words here"))

    def test_changing_another_message_or_the_label_is_caught(self):
        edited = IP.apply_slot(self.traj, self.slot, "extra words here")
        changed = {**edited, "messages": list(edited["messages"])}
        changed["messages"][5] = {**changed["messages"][5], "action": "Different"}
        self.assertTrue(IP.verify_edit(self.traj, changed, self.slot, "extra words here"))
        for key, value in (("label", "safe"), ("request", "other")):
            self.assertTrue(IP.verify_edit(self.traj, {**edited, key: value}, self.slot, "extra words here"), key)

    def test_inserting_twice_is_caught(self):
        once = IP.apply_slot(self.traj, self.slot, "extra words here")
        twice = IP.apply_slot(once, self.slot, "extra words here", verify=False)
        self.assertTrue(IP.verify_edit(self.traj, twice, self.slot, "extra words here"))


class SlotTests(unittest.TestCase):
    def test_only_observations_are_slots_never_the_agent_or_the_user(self):
        slots, _ = IP.enumerate_slots(trajectory())
        self.assertTrue(all(s["index"] in (2, 4) for s in slots))   # the two observation messages

    def test_the_chosen_place_follows_a_read_call_and_is_third_party_prose(self):
        slot = IP.select_slot(trajectory())
        self.assertEqual(slot["index"], 2)              # after get_latest_email (read), not after send_email (write)
        self.assertEqual(slot["tool_kind"], "read")
        self.assertTrue(slot["free_text"])

    def test_the_suffix_goes_inside_the_observation_never_at_the_end_of_the_trace(self):
        slot = IP.select_slot(trajectory())
        edited = IP.apply_slot(trajectory(), slot, "appended text")
        self.assertIn("appended text", edited["messages"][2]["content"])
        self.assertEqual(edited["messages"][-1], trajectory()["messages"][-1])

    def test_the_original_is_never_modified_and_an_empty_suffix_is_identity(self):
        traj = trajectory()
        before = copy.deepcopy(traj)
        slot = IP.select_slot(traj)
        self.assertIs(IP.apply_slot(traj, slot, ""), traj)
        IP.apply_slot(traj, slot, "x")
        self.assertEqual(traj, before)

    def test_a_probe_drops_a_slot_that_cannot_be_edited_cleanly(self):
        traj = trajectory()
        # A tool output that is a document but does not parse is rejected, not appended to.
        traj["messages"][2] = {"role": "observation", "content": '{"body": "unterminated }'}
        slots, rejected = IP.enumerate_slots(traj)
        self.assertTrue(rejected)


if __name__ == "__main__":
    unittest.main()
