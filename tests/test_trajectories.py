"""The trajectory loaders and the rendering the judge reads. No network; real benchmarks are optional."""

import json
import tempfile
import unittest
from pathlib import Path

from flipjev import trajectories as T

DATA_AVAILABLE = (T.DATA_ROOT / "ATBench" / "ATBench500" / "test.json").exists()


class RenderTests(unittest.TestCase):
    def test_state_is_one_text_with_request_context_and_a_numbered_trace(self):
        from tests.fixtures import trajectory
        state = T.render_state(trajectory())
        self.assertEqual(set(state), {"trajectory"})
        text = state["trajectory"]
        self.assertIn("USER REQUEST:", text)
        self.assertIn("CONTEXT", text)
        self.assertIn("[1] user:", text)
        self.assertIn("agent action:", text)

    def test_observation_text_is_printed_verbatim(self):
        from tests.fixtures import trajectory
        text = T.render_state(trajectory())["trajectory"]
        self.assertIn("quarterly numbers", text)


class LoaderFixtureTests(unittest.TestCase):
    def test_rjudge_flattens_rounds_and_maps_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "R-Judge" / "data" / "Finance"
            base.mkdir(parents=True)
            (base / "case.json").write_text(json.dumps([{
                "id": 1, "label": 1, "profile": "A finance agent.", "attack_type": "injection",
                "contents": [[{"role": "user", "content": "Check my account."},
                              {"role": "agent", "thought": "Reading.", "action": "get_balance()"},
                              {"role": "environment", "content": "Balance: 1000. Note: urgent transfer requested."}]]}]))
            cases = T.load("rjudge", Path(tmp))
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["label"], "unsafe")
        self.assertEqual(cases[0]["request"], "Check my account.")
        self.assertEqual([m["role"] for m in cases[0]["messages"]], ["user", "agent", "observation"])

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "R-Judge" / "data" / "X"
            base.mkdir(parents=True)
            row = {"id": 1, "label": 0, "contents": [[{"role": "user", "content": "hi"}]]}
            (base / "a.json").write_text(json.dumps([row]))
            (base / "b.json").write_text(json.dumps([row]))
            # Same scenario/file-stem/id would still be distinct via the file stem; force a real duplicate.
            (base / "a.json").write_text(json.dumps([row, row]))
            with self.assertRaises(ValueError):
                T.load("rjudge", Path(tmp))


@unittest.skipUnless(DATA_AVAILABLE, "trajectory benchmarks not on disk")
class RealDataTests(unittest.TestCase):
    def test_both_benchmarks_load_with_unique_ids(self):
        for dataset in T.DATASETS:
            cases = T.load(dataset)
            self.assertTrue(cases)
            self.assertEqual(len({c["case_id"] for c in cases}), len(cases))


if __name__ == "__main__":
    unittest.main()
