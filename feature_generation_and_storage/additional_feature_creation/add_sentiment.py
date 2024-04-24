import nltk

nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

from feature_generation_and_storage.additional_feature_creation.add_new_feature_abstract_class import AddNewFeature
from feature_generation_and_storage.sentence_model import Sentence


class AddSentiment(AddNewFeature):
    def update_sentence_object(self, sentence: Sentence) -> None:
        sia = SentimentIntensityAnalyzer()
        scores: dict = sia.polarity_scores(sentence.sentence_content)

        sentence.positive_vader_score = scores["pos"]
        sentence.negative_vader_score = scores["neg"]
        sentence.neutral_vader_score = scores["neu"]
        sentence.compound_vader_score = scores["compound"]
