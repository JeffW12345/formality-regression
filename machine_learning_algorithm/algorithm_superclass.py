import os
from abc import ABC, abstractmethod

import numpy as np
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

        self.rmse_scores = []
        self.r2_scores = []

    @abstractmethod
    def train_test_and_publish(self) -> None:
        pass

    @abstractmethod
    def publish_results(self) -> None:
        pass

    def update_mean_squared_error_and_r_squared_in_results_object(self):
        self.results.root_mean_squared_error = np.mean(self.rmse_scores)
        self.results.r_squared = np.mean(self.r2_scores)
