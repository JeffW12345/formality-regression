import pandas as pd

from feature_generation_and_storage.add_new_feature.add_parts_of_speech import AddPartsOfSpeech
from feature_generation_and_storage.add_new_feature.add_sentiment import AddSentiment
from feature_generation_and_storage.add_new_feature.add_spelling_and_grammar_count import AddSpellingAndGrammarCount
from feature_generation_and_storage.add_new_feature.add_syllable_count import AddSyllableCount
from feature_generation_and_storage.import_source_data import import_data_and_store_sentence_objects
from feature_generation_and_storage.sentence_storage import sentence_store


def create_feature_and_target_file():
    import_data_and_store_sentence_objects()

    list_of_add_new_feature_subclasses = [
        AddPartsOfSpeech(),
        AddSentiment(),
        AddSpellingAndGrammarCount(),
        AddSyllableCount()
    ]

    for sentence in sentence_store:
        for add_new_feature_subclass in list_of_add_new_feature_subclasses:
            add_new_feature_subclass.update_sentence_object(sentence)

    file_path = "csv_files/complete_data.csv"

    complete_data_df = pd.DataFrame([vars(match) for match in sentence_store])

    complete_data_df.to_csv(file_path, index=False)

create_feature_and_target_file()