import unittest
from unittest.mock import patch
from feature_generation_and_storage.add_new_feature.add_spelling_and_grammar_count import AddSpellingAndGrammarCount
from feature_generation_and_storage.sentence import Sentence

class TestAddSpellingAndGrammarCount(unittest.TestCase):
    def setUp(self):
        self.add_feature = AddSpellingAndGrammarCount()

    @patch('feature_generation_and_storage.add_new_feature.add_spelling_and_grammar_count.language_tool_python.LanguageTool')
    def test_update_sentence_object_with_no_errors(self, mock_tool):
        mock_tool.return_value.check.return_value = []
        sentence_content = "This is a test sentence with no errors."
        sentence = Sentence(sentence_content=sentence_content)
        sentence.length_in_characters = 43

        self.add_feature.update_sentence_object(sentence)

        self.assertEqual(0, sentence.number_of_spelling_and_grammatical_errors)
        self.assertEqual(0.0, sentence.spelling_and_grammar_errors_per_character)

    @patch('feature_generation_and_storage.add_new_feature.add_spelling_and_grammar_count.language_tool_python.LanguageTool')
    def test_update_sentence_object_with_errors(self, mock_tool):
        mock_tool.return_value.check.return_value = [{'message': 'Spelling mistake'}]
        sentence_content = "This is a test sentense with an error."
        sentence = Sentence(sentence_content=sentence_content)
        sentence.length_in_characters = 41

        self.add_feature.update_sentence_object(sentence)

        self.assertEqual(1, sentence.number_of_spelling_and_grammatical_errors)
        self.assertEqual(0.024, sentence.spelling_and_grammar_errors_per_character)

if __name__ == '__main__':
    unittest.main()
