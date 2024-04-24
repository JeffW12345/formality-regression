import nltk

nltk.download("stopwords")

from feature_generation_and_storage.add_new_feature.add_new_feature_abstract_class import AddNewFeature
from feature_generation_and_storage.sentence_model import Sentence

from nltk.corpus import stopwords

import syllables


class AddSyllableCount(AddNewFeature):
    def update_sentence_object(self, sentence: Sentence) -> None:
        words_in_sentence: list = sentence.sentence_content.split(" ")

        self.update_syllable_data_stop_words_included(sentence, words_in_sentence)

        self.update_syllable_data_stop_words_excluded(sentence, words_in_sentence)

    def update_syllable_data_stop_words_included(self, sentence: Sentence, words_in_sentence: list) -> None:
        total_syllables_stop_words_included = 0
        for word in words_in_sentence:
            total_syllables_stop_words_included += syllables.estimate(word)

        sentence.total_syllables_stop_words_included = total_syllables_stop_words_included
        sentence.total_syllables_per_word_stop_words_included = \
            total_syllables_stop_words_included / sentence.length_in_words

    def update_syllable_data_stop_words_excluded(self, sentence: Sentence, words_in_sentence: list) -> None:
        stop_words = set(stopwords.words("english"))
        sentence_without_stop_words: list = [word for word in words_in_sentence if word.casefold() not in stop_words]

        total_syllables_stop_words_excluded = 0
        for word in sentence_without_stop_words:
            total_syllables_stop_words_excluded += syllables.estimate(word)

        sentence.total_syllables_stop_words_excluded = total_syllables_stop_words_excluded
        sentence.total_syllables_per_word_stop_words_excluded = \
            total_syllables_stop_words_excluded / len(sentence_without_stop_words)



