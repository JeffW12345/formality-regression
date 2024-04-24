import nltk

nltk.download("stopwords")

from feature_generation_and_storage.additional_feature_creation.add_new_feature_abstract_class import AddNewFeature
from feature_generation_and_storage.sentence_model import Sentence

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class AddPartsOfSpeech(AddNewFeature):
    def update_sentence_object(self, sentence: Sentence) -> None:
        words_in_sentence: list = word_tokenize(sentence.sentence_content)

        number_of_nouns = 0
        number_of_adjectives = 0
        number_of_adverbs = 0
        number_of_verbs = 0
        number_of_pronouns = 0

        pos_tags = nltk.pos_tag(words_in_sentence)

        for _, part_of_word in pos_tags:
            if "NN" in part_of_word:
                number_of_nouns += 1
            elif "JJ" in part_of_word:
                number_of_adjectives += 1
            elif "RB" in part_of_word:
                number_of_adverbs += 1
            elif "VB" in part_of_word:
                number_of_verbs += 1
            elif "PRP" in part_of_word:
                number_of_pronouns += 1

        sentence.number_of_nouns = number_of_nouns
        sentence.number_of_adjectives = number_of_adjectives
        sentence.number_of_adverbs = number_of_adverbs
        sentence.number_of_verbs = number_of_verbs
        sentence.number_of_pronouns = number_of_pronouns

        self.update_with_number_of_stop_words(sentence, words_in_sentence)
        self.update_with_proportion_of_stop_words(sentence)


    def update_with_number_of_stop_words(self, sentence: Sentence, words_in_sentence: list) -> None:
        stop_words = set(stopwords.words("english"))
        sentence_with_only_stop_words: list = [word for word in words_in_sentence if word.casefold() in stop_words]
        sentence.number_of_stop_words = len(sentence_with_only_stop_words)

    def update_with_proportion_of_stop_words(self, sentence):
        sentence.proportion_of_stop_words = round(sentence.number_of_stop_words / sentence.length_in_words, 3)
