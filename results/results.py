import os
from string import capwords

import pandas as pd


class Results:
    def __init__(self):
        self.algorithm_name = None
        self.root_mean_squared_error = None
        self.r_squared = None
        self.mean_absolute_error = None

    def print_to_spreadsheet(self):
        self._set_os_to_current_directory()
        filename = "results.csv"
        df = pd.DataFrame([vars(self)])

        self._round_dataframe_to_three_decimal_places(df)

        self._format_headers(df)

        if os.path.exists(filename):
            df.to_csv(filename, index=False, mode='a', header=False)
        else:
            df.to_csv(filename, index=False)


    def _round_dataframe_to_three_decimal_places(self, df):
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].round(3)

    def _set_os_to_current_directory(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir(current_dir)

    def _format_headers(self, df):
        df.columns = [capwords(col.replace('_', ' ')) for col in df.columns]
