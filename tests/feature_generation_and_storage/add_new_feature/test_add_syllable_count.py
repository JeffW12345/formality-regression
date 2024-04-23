import unittest
from unittest.mock import MagicMock

from nltk.corpus import stopwords

from feature_generation_and_storage.add_new_feature.add_syllable_count import AddSyllableCount
from feature_generation_and_storage.sentence import Sentence


class TestAddSyllableCount(unittest.TestCase):
    def setUp(self):
        self.add_syllable_count = AddSyllableCount()

    def test_update_syllable_data_stop_words_included(self):
        sentence = Sentence()
        sentence.sentence_content = "The quick brown fox jumps over the lazy dog"
        words_in_sentence = sentence.sentence_content.split(" ")
        sentence.length_in_words = 9

        self.add_syllable_count.update_syllable_data_stop_words_included(sentence, words_in_sentence)

        self.assertEqual(11, sentence.total_syllables_stop_words_included)
        self.assertAlmostEqual(11 / 9, sentence.total_syllables_per_word_stop_words_included)

    def test_update_syllable_data_stop_words_excluded(self):
        sentence = Sentence()
        sentence.sentence_content = "The quick brown fox jumps over the lazy dog"
        words_in_sentence = sentence.sentence_content.split()

        stopwords.words = MagicMock(return_value={"the", "over"})

        self.add_syllable_count.update_syllable_data_stop_words_excluded(sentence, words_in_sentence)

        self.assertEqual(7, sentence.total_syllables_stop_words_excluded)
        self.assertAlmostEqual(7/6, sentence.total_syllables_per_word_stop_words_excluded)

if __name__ == "__main__":
    unittest.main()
