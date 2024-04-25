from abc import ABC, abstractmethod

import pandas as pd
from pandas import DataFrame

from results.results import Results


class MachineLearningAlgorithm(ABC):
    def __init__(self):
        self.results: Results = Results()
        self.df: DataFrame = pd.read_csv(r"..\csv_files\complete_data.csv")
        self.X = self.df.drop(['formality_score_from_raters', 'sentence_content'], axis=1).to_numpy()
        self.y = self.df['formality_score_from_raters'].to_numpy()

    @abstractmethod
    def train_and_test(self) -> None:
        pass