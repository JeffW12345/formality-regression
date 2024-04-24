from feature_generation_and_storage.add_new_feature.add_new_feature_abstract_class import AddNewFeature
from feature_generation_and_storage.sentence_model import Sentence

import language_tool_python


class AddSpellingAndGrammarCount(AddNewFeature):
    def update_sentence_object(self, sentence: Sentence) -> None:
        tool = language_tool_python.tool = language_tool_python.LanguageToolPublicAPI('en-US')
        errors = tool.check(sentence.sentence_content)

        sentence.number_of_spelling_and_grammatical_errors = len(errors)
        sentence.spelling_and_grammar_errors_per_character = round(float(len(errors) / sentence.length_in_characters), 3)
