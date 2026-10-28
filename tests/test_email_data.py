"""The fixed email sets and the long emails are built correctly and keep the correct label. No network."""

import unittest

from flipjev.email_sets import email_sets, names_subject
from flipjev.long_emails import check, long_email_cases


class EmailSetTests(unittest.TestCase):
    def setUp(self):
        self.sets = email_sets()

    def test_every_set_is_balanced_and_matches_its_case_count(self):
        for name, (cases, meta) in self.sets.items():
            self.assertEqual(len(cases), meta["case_count"], name)
            clean = [c for c in cases if c["kind"] == "clean"]
            self.assertEqual(sum(c["label"] == "yes" for c in clean), sum(c["label"] == "no" for c in clean), name)

    def test_no_injection_names_the_subject(self):
        for name, (cases, _) in self.sets.items():
            for case in cases:
                if case.get("addition"):
                    self.assertFalse(names_subject(case["addition"]), (name, case["case_id"]))

    def test_injections_are_paired_with_a_control(self):
        cases, _ = self.sets["context_ignoring"]
        by_id = {c["case_id"]: c for c in cases}
        for case in cases:
            if case["kind"] == "injection":
                self.assertIn(case["control_case_id"], by_id, case["case_id"])
                self.assertEqual(by_id[case["control_case_id"]]["kind"], "control")

    def test_hard_set_has_the_three_difficulties_and_a_fake_answer(self):
        cases, meta = self.sets["hard_emails"]
        self.assertEqual(meta["base_emails"], {"easy": 40, "hard": 32, "long": 8})
        fake = next(c for c in cases if c["variant"].startswith("fake_output") and c["kind"] == "injection")
        self.assertIn("probabilities", fake["addition"])


class LongEmailTests(unittest.TestCase):
    def test_twenty_emails_with_a_middle_mention_only_in_yes(self):
        cases = long_email_cases()
        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(c["label"] == "yes" for c in cases), 10)
        check(cases)   # raises if a decisive sentence is not in a middle paragraph, or a no email mentions the course


if __name__ == "__main__":
    unittest.main()
