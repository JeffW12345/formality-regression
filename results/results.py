import os
from string import capwords

import pandas as pd


def _round_dataframe_to_three_decimal_places(df) -> None:
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].round(3)


def _set_directory_to_current_location() -> None:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(current_dir)


def _format_headers(df) -> None:
    df.columns = [capwords(col.replace('_', ' ')) for col in df.columns]


class Results:
    def __init__(self):
        self.algorithm_name = None
        self.has_normalisation = False
        self.has_standardisation = False
        self.root_mean_squared_error = None
        self.r_squared = None

    def print_to_spreadsheet(self) -> None:
        _set_directory_to_current_location()
        filename = "results.csv"
        df = pd.DataFrame([vars(self)])

        _round_dataframe_to_three_decimal_places(df)

        _format_headers(df)

        df.to_csv(filename, index=False)
