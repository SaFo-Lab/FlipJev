"""The dataset registry and task specifications. No network (no documents are loaded)."""

import unittest

from flipjev import datasets as D


class RegistryTests(unittest.TestCase):
    def test_nine_topic_datasets_and_imdb(self):
        self.assertEqual(len(D.TOPIC_DATASETS), 9)
        self.assertIn("imdb", D.REGISTRY)
        self.assertNotIn("imdb", D.TOPIC_DATASETS)

    def test_every_dataset_has_a_question_and_a_source(self):
        for name, config in D.REGISTRY.items():
            self.assertTrue(config["question"].endswith("?"), name)
            self.assertTrue(config["source"], name)

    def test_task_for_builds_a_spec_whose_forbidden_words_block_the_subject(self):
        spec = D.task_for("ag_news")
        self.assertEqual(spec.primary, "plain_noul")
        self.assertEqual(spec.target, "no")
        self.assertFalse(spec.allowed("this is about a football team"))
        self.assertTrue(spec.allowed("question: is 'no' accurate"))

    def test_the_choice_primary_is_selectable(self):
        self.assertEqual(D.task_for("trec", primary="plain_choice").primary, "plain_choice")

    def test_imdb_forbids_nothing_because_sentiment_is_not_label_preserving(self):
        self.assertEqual(D.REGISTRY["imdb"]["forbidden"], [])


if __name__ == "__main__":
    unittest.main()
