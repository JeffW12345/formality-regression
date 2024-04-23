import unittest

from feature_generation_and_storage.add_new_feature.add_parts_of_speech import AddPartsOfSpeech
from feature_generation_and_storage.sentence import Sentence


class TestAddPartsOfSpeech(unittest.TestCase):
    def setUp(self):
        self.add_parts_of_speech = AddPartsOfSpeech()

    def test_update_sentence_object(self):
        sentence = Sentence()
        sentence.sentence_content = \
            "She said, 'A spoonful of sugar helps the medicine go down, and in a most delightful way.'"
        self.add_parts_of_speech.update_sentence_object(sentence)

        self.assertEqual(4, sentence.number_of_nouns)
        self.assertEqual(1, sentence.number_of_adjectives)
        self.assertEqual(2, sentence.number_of_adverbs)
        self.assertEqual(3, sentence.number_of_verbs)
        self.assertEqual(1, sentence.number_of_pronouns)
        self.assertEqual(9, sentence.number_of_stop_words)

    def test_update_with_number_of_stop_words(self):
        sentence = Sentence()
        sentence.sentence_content = "The quick brown fox jumps over the lazy dog"
        words_in_sentence = sentence.sentence_content.split()

        self.add_parts_of_speech.update_with_number_of_stop_words(sentence, words_in_sentence)
        self.assertEqual(sentence.number_of_stop_words, 3)


if __name__ == "__main__":
    unittest.main()
