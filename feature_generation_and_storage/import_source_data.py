import os
import pandas as pd

from feature_generation_and_storage.sentence import Sentence
from feature_generation_and_storage.sentence_storage import sentence_store


def import_data_and_store_sentence_objects():
    file_path = "csv_files/original_formality_dataset.csv"

    if os.path.exists(file_path):
        original_data_dataframe = pd.read_csv(file_path)
    else:
        raise FileExistsError("File not present at location specified")

    original_data_dataframe = original_data_dataframe.drop_duplicates()

    original_data_dataframe = original_data_dataframe.dropna(subset=['Actual sentence', 'Formality'])

    for index, row in original_data_dataframe.iterrows():
        sentence_id = row['Sentence ID']
        formality_score_from_raters = row['Formality']
        informativeness = row['Informativeness']
        implicature_score = row['Implicature']
        length_in_words = row['Length in Words']
        length_in_characters = row['Length in Characters']
        f_score = row['F-score']
        i_score = row['I-score']
        lexical_density = row['Lexical Density']
        sentence_content = row['Actual sentence']

        sentence = Sentence(sentence_id=sentence_id,
                            formality_score_from_raters=formality_score_from_raters,
                            informativeness=informativeness,
                            implicature_score=implicature_score,
                            length_in_words=length_in_words,
                            length_in_characters=length_in_characters,
                            f_score=f_score,
                            i_score=i_score,
                            lexical_density=lexical_density,
                            sentence_content=sentence_content)

        sentence_store.add(sentence)

    print(f"Imported {len(sentence_store)} records")