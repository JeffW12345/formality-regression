import os
from abc import ABC, abstractmethod

import pandas as pd
from pandas import DataFrame

from results.results import Results


def _set_directory_to_current_folder() -> None:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(current_dir)


class MachineLearningAlgorithm(ABC):
    def __init__(self):
        _set_directory_to_current_folder()
        self.results: Results = Results()
        self.df: DataFrame = pd.read_csv(r"..\source_target_and_feature_csv_files\complete_data.csv")
        self.X = self.df.drop(['formality_score_from_raters', 'sentence_content'], axis=1).to_numpy()
        self.y = self.df['formality_score_from_raters'].to_numpy()

    @abstractmethod
    def train_test_and_publish(self) -> None:
        pass
