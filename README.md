INTRODUCTION
============

_"Hey, why don't you come along?"_

_"You are cordially invited to attend."_

The above sentences mean the same thing, but have vastly different levels of formality. 

This app takes a corpus of 7,000 sentences which were given formality scores ranging from 1 (highly informal) to 7 
(highly formal) by human raters. It uses machine learning to attempt to predict the scores given by the human raters.

It builds on a university product used I carried out explore whether a machine learning program could successfully 
predict whether humans regarded a sentence as formal or informal (it can be viewed [here](https://github.com/JeffW12345/formality-classification-using-machine-learning )).

PROJECT STRUCTURE
=================

**Additional feature generation and storage**

The _feature_generation_and_storage_ package contains code which imports a CSV containing the original 7,000 sentences 
and data about those sentences (such as number of characters). It derives additional features from the sentences, 
relating to things such as the number of nouns and verbs in search sentence, the number of spelling and grammar error per 
character in the sentence, and how emotionally charged the content is. 

The additional feature generation and collation is controlled by the data_collection_controller.py module. Running that
module's 'create_feature_and_target_file()' function results in the creation of a csv file containing the additional 
features. 

**CSV files containing features and target**

The original CSV file and the CSV file with the additional features (called complete_data.csv) are stored in the 
'source_target_and_feature_csv_files' directory.

**Exploring the data**

The data_summaries package contains an ExploreDataRelationships class which has methods that do the following:

- Produce a spreadsheet that provides correlations data for the features and the target data, making it possible to 
explore correlations between features and between individual features and the target (the sentence ratings).

- Produce a spreadsheet that summarises each of the feature fields and the target fields, in terms of things like mean 
and standard deviation. 

- Generate a histogram relating to the target or to any feature field. 

- Generate a 'feature to target' scatter graph for any feature.

**Model building and publishing**

The _machine_learning_algorithm_ package contains modules which train and test the data, and then publish the results in 
a shared file, results.csv (which is located in the 'results' folder). Each of these modules extends the 
MachineLearningAlgorithm class, which is an abstract class with an abstract method of 'train_test_and_publish()'. 

The 'main' class in the root folder iterates through a list of objects of type MachineLearningAlgorithm, and runs each
object's 'train_test_and_publish()' method in turn. 

RUNNING THE TESTS
=================

The tests can be run by running the 'main' module. Doing so will result in the 'results.csv' csv file in the 'results' 
directory being updated with the latest results.

A requirements.txt file is included in the repository. 