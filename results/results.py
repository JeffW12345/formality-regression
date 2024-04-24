import os

import pandas as pd


class Results:
    def __init__(self):
        self.classifier_name = None
        self.root_mean_squared_error = None
        self.r_squared = None
        self.mean_absolute_error = None

    def print_to_spreadsheet(self):
        filename = "results.csv"
        df = pd.DataFrame([vars(self)])

        if os.path.exists(filename):
            df.to_csv(filename, index=False, mode='a', header=False)
        else:
            df.to_csv(filename, index=False)
