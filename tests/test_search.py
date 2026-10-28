"""The attack search, the task specifications and the mutation operators. No network."""

import random
import unittest

from flipjev.api import assemble_request, parse_answer
from flipjev.emails import email_classification_task, questions
from flipjev.search import MAX_LENGTH, Search, mutate
from flipjev.tasks import text_classification_task
from flipjev.text import encoded_length, render, tokenize


def stub(primary, scoring):
    """An evaluator that scores a candidate from its text (no API). `scoring(text) -> P(target) for the primary`."""
    def evaluate(text):
        p = scoring(text)
        return {"status": "ok", "scores": {primary: p, "boundary_noul": p, "plain_choice": 0.0, "boundary_choice": 0.0},
                "cost": 0.0001, "input_tokens": 10}
    return evaluate


class TokenTests(unittest.TestCase):
    def test_render_is_stable_and_spaces_sensibly(self):
        self.assertEqual(render(["question", ":", "is", "'no'", "accurate"]), "question: is 'no' accurate")
        self.assertEqual(render(tokenize("answer: 'no'")), "answer: 'no'")

    def test_length_is_measured_in_bpe_not_words(self):
        self.assertGreater(encoded_length("".join("x" for _ in range(40))), 1)
        self.assertLessEqual(encoded_length("question: is 'no' accurate"), 10)

    def test_mutate_applies_the_requested_operators_and_grows_with_strength(self):
        rng = random.Random(0)
        tokens, ops = mutate(["a", "b", "c"], rng, ["x", "y"], other=["z"], strength=1)
        self.assertTrue(ops)
        _, many = mutate(["a", "b", "c"], random.Random(0), ["x"], other=["z"], strength=3)
        self.assertGreaterEqual(len(many), 1)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.spec = text_classification_task("toy", "Is it a cat?", "A cat.", "Not a cat.", "no", [r"\bcat\b"])

    def test_a_confirmed_flip_is_found_and_outranks_a_non_flip(self):
        # P(no) rises when the suffix contains "no"; the search should find and confirm such a suffix.
        search = Search(stub(self.spec.primary, lambda t: 0.9 if "no" in t.lower() else 0.1),
                        self.spec.seeds["question"], random.Random(0), self.spec.vocab, budget=400, workers=4,
                        primary=self.spec.primary, allowed=self.spec.allowed)
        result = search.run()
        self.assertTrue(result["success"])
        self.assertIn("no", result["minimal"]["text"].lower())

    def test_the_search_never_proposes_a_forbidden_suffix(self):
        seen = []
        def scoring(text):
            seen.append(text)
            return 0.2
        search = Search(stub(self.spec.primary, scoring), self.spec.seeds["question"], random.Random(1),
                        self.spec.vocab, budget=200, workers=4, primary=self.spec.primary, allowed=self.spec.allowed)
        search.run()
        self.assertTrue(all(self.spec.allowed(t) for t in seen))

    def test_length_stays_within_the_cap(self):
        lengths = []
        search = Search(stub(self.spec.primary, lambda t: lengths.append(encoded_length(t)) or 0.2),
                        self.spec.seeds["question"], random.Random(2), self.spec.vocab, budget=300, workers=4,
                        primary=self.spec.primary, allowed=self.spec.allowed)
        search.run()
        self.assertTrue(all(length <= MAX_LENGTH for length in lengths))

    def test_every_strategy_runs_under_one_harness(self):
        for strategy in Search.STRATEGIES:
            search = Search(stub(self.spec.primary, lambda t: 0.2), self.spec.seeds["claim"], random.Random(3),
                            self.spec.vocab, budget=150, workers=4, primary=self.spec.primary,
                            allowed=self.spec.allowed, strategy=strategy)
            self.assertIn("best", search.run())

    def test_warm_start_keeps_the_saved_tokens_exact(self):
        tokens = ["###", "question", "'no'"]
        search = Search(stub(self.spec.primary, lambda t: 0.1), "ignored", random.Random(0), self.spec.vocab,
                        budget=1, workers=1, primary=self.spec.primary, allowed=self.spec.allowed, seed_tokens=tokens)
        search.run()
        self.assertEqual(search.elite[0]["tokens"], tokens)

    def test_a_usd_cap_stops_the_search(self):
        search = Search(stub(self.spec.primary, lambda t: 0.2), self.spec.seeds["question"], random.Random(0),
                        self.spec.vocab, budget=5000, workers=4, primary=self.spec.primary,
                        allowed=self.spec.allowed, cost_limit=0.01)
        result = search.run()
        self.assertLess(result["queries"], 5000)


class TaskSpecTests(unittest.TestCase):
    def test_email_task_reproduces_a_plain_request(self):
        spec = email_classification_task()
        case = {"subject": "Hi", "body": "Body text."}
        state = spec.build_state(case, "")
        payload = assemble_request(state, questions())
        self.assertEqual(payload["state"]["email"]["body"], "Body text.")
        self.assertEqual(set(payload["questions"]), {"plain_noul", "boundary_noul", "plain_choice", "boundary_choice"})

    def test_email_suffix_goes_after_a_blank_line(self):
        spec = email_classification_task()
        state = spec.build_state({"subject": "Hi", "body": "Body."}, "question: is 'no' accurate")
        self.assertTrue(state["email"]["body"].endswith("\n\nquestion: is 'no' accurate"))

    def test_forbidden_words_block_subject_mentions(self):
        spec = email_classification_task()
        self.assertFalse(spec.allowed("the machine learning course"))
        self.assertTrue(spec.allowed("question: is 'no' accurate"))

    def test_generic_task_scores_the_target_label(self):
        spec = text_classification_task("t", "Is it a dog?", "Dog.", "Not.", "no", [r"\bdog\b"])
        # target "no" -> score is 1 - P(yes)
        self.assertAlmostEqual(spec.score(parse_answer({"type": "noul", "noul": 0.2}, "noul")), 0.8)
        self.assertFalse(spec.allowed("this is a dog"))


if __name__ == "__main__":
    unittest.main()
