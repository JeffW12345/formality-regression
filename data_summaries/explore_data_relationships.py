import os
import matplotlib.pyplot as plt
import pandas as pd

from pandas import DataFrame


class ExploreDataRelationships:
    def __init__(self):
        self._set_directory_to_current_directory()

        self.all_fields_except_sentences: DataFrame = pd.read_csv(r"..\csv_files\complete_data.csv") \
            .drop(['sentence_content', 'sentence_id'], axis=1)

        self.features: DataFrame = pd.read_csv(r"..\csv_files\complete_data.csv") \
            .drop(['formality_score_from_raters', 'sentence_content', 'sentence_id'], axis=1)

        self.targets: DataFrame = pd.read_csv(r"..\csv_files\complete_data.csv") \
            [['formality_score_from_raters']]

    def write_correlations_to_file(self):
        correlations = self.all_fields_except_sentences.corr()
        correlations.to_csv("correlations.csv")

    def field_statistical_summaries(self):
        field_summaries = self.all_fields_except_sentences.describe()
        field_summaries.to_csv("field_summaries.csv")

    def generate_histogram_for_field(self, bins=40, field="formality_score_from_raters"):
        plt.hist(self.all_fields_except_sentences[field], bins=bins)
        plt.show()

    def generate_feature_vs_target_scatter_graph(self, feature_field_heading):
        combined_df = pd.concat([self.targets, self.features[feature_field_heading]], axis=1)
        scatter_graph = combined_df.plot(
            kind='scatter',
            x=feature_field_heading,
            y="formality_score_from_raters",
            title=f"{feature_field_heading} vs formality ratings"
        )
        plt.show()

    def _set_directory_to_current_directory(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir(current_dir)