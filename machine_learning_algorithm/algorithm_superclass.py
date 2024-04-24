from abc import ABC, abstractmethod

from pandas import DataFrame

from results.results import Results


class MachineLearningAlgorithm(ABC):
    def __init__(self):
        self.results: Results = Results()
        self.train_X_y: tuple = ()

    @abstractmethod
    def co_ordinate_actions(self) -> None:
        pass

    @abstractmethod
    def import_data_from_file(self) -> DataFrame:
        pass

    @abstractmethod
    def process_data(self) -> None:
        pass

    @abstractmethod
    def publish_results(self, ) -> None:
        pass

    @abstractmethod
    def visualise_data(self) -> None:
        pass