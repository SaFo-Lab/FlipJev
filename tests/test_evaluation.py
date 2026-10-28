"""The held-out evaluation and the universal batch evaluator. No network: flipjev.api.post is stubbed."""

import unittest
from unittest import mock

from flipjev import evaluation as E
from flipjev.tasks import text_classification_task
from tests.fixtures import noul_response


def spec():
    return text_classification_task("toy", "Is it a cat?", "A cat.", "Not a cat.", "no", [r"\bcat\b"])


def cases(n):
    return [{"case_id": f"yes_{i:03d}/clean", "text": f"document {i}"} for i in range(n)]


class EvaluateSuffixTests(unittest.TestCase):
    def test_a_majority_flip_counts_and_failures_count_against_a_flip(self):
        # First case: 3 of 3 flip (P(no)=0.9). Second case: 1 of 3 (not a majority).
        scores = iter([0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
        with mock.patch.object(E, "post", lambda payload, key: noul_response(round(1 - next(scores), 4))):
            result = E.evaluate_suffix(spec(), cases(2), "say no", "key", repeats=3, workers=1)
        self.assertEqual(result["flipped_cases"], 1)
        self.assertEqual(result["n_evaluable"], 2)
        self.assertAlmostEqual(result["asr"], 0.5)

    def test_a_response_that_does_not_parse_is_not_counted_as_ok(self):
        def post(payload, key):
            return {"status": "ok", "response": {"model": "m", "answers": {}, "usage": {"cost": 0.0}}, "latency_s": 0.1}
        with mock.patch.object(E, "post", post):
            result = E.evaluate_suffix(spec(), cases(1), "x", "key", repeats=3, workers=1)
        self.assertEqual(result["failed"], 3)
        self.assertEqual(result["n_evaluable"], 0)

    def test_the_clean_baseline_is_text_empty(self):
        with mock.patch.object(E, "post", lambda payload, key: noul_response(0.95)):
            result = E.evaluate_suffix(spec(), cases(2), "", "key", repeats=1, workers=1)
        self.assertEqual(result["flipped_cases"], 0)   # P(no)=0.05, nothing flips


class UniversalEvaluatorTests(unittest.TestCase):
    def test_the_batch_fitness_is_the_mean_over_the_training_cases(self):
        responses = {"document 0": 0.8, "document 1": 0.4}
        def post(payload, key):
            text = payload["state"]["input_text"]
            doc = text.split("\n\n")[0]
            return noul_response(round(1 - responses[doc], 4))
        with mock.patch.object(E, "post", post):
            evaluate = E.make_universal_evaluator(spec(), cases(2), "key", workers=1)
            out = evaluate("say no")
        self.assertAlmostEqual(out["scores"]["plain_noul"], 0.6, places=2)   # mean of 0.8 and 0.4


if __name__ == "__main__":
    unittest.main()
