from abc import ABC, abstractmethod

from feature_generation_and_storage.sentence_model import Sentence


class AddNewFeature(ABC):
    """
    Abstract class to define the structure of adding a new feature.
    """

    @abstractmethod
    def update_sentence_object(self, sentence: Sentence) -> None:
        pass