import unittest

from feature_generation_and_storage.add_new_feature.add_sentiment import AddSentiment
from feature_generation_and_storage.sentence import Sentence


class TestAddSentiment(unittest.TestCase):
    def setUp(self):
        self.add_sentiment = AddSentiment()

    # Expected figures from https://github.com/cjhutto/vaderSentiment
    def test_correct_scores(self):
        sentence = Sentence()
        sentence.sentence_content = "VADER is smart, handsome, and funny."

        self.add_sentiment.update_sentence_object(sentence)

        self.assertEqual(0.746, sentence.positive_vader_score)
        self.assertEqual(0.8316, sentence.compound_vader_score)
        self.assertEqual(0.254, sentence.neutral_vader_score)
        self.assertEqual(0.0, sentence.negative_vader_score)
