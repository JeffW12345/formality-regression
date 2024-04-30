import os
from abc import ABC, abstractmethod

import pandas as pd
from pandas import DataFrame

from results.results import Results

class MachineLearningAlgorithm(ABC):
    def __init__(self):
        self._set_os_to_current_directory()
        self.results: Results = Results()
        self.df: DataFrame = pd.read_csv(r"..\source_target_and_feature_csv_files\complete_data.csv")
        self.X = self.df.drop(['formality_score_from_raters', 'sentence_content'], axis=1).to_numpy()
        self.y = self.df['formality_score_from_raters'].to_numpy()

    def _set_os_to_current_directory(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir(current_dir)

    @abstractmethod
    def train_test_and_publish(self) -> None:
        pass