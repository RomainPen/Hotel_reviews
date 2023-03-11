# Hotel Review Analysis

This project aims to analyze hotel reviews and determine the strong and weak points of each hotel. The goal is to create an API that can provide this information to hotels, so they can improve and address any issues identified in the reviews. The data used in this project was webscrapped from Tripadvisor.


**********************************************************************************************************************************************************
The `Rendu_final` file  contain : 
- the report `Rapport_GEFFLOT_PENICHON_GHAZALI_AIYAD_MENANI.ipynb`
- the production `Production_GEFFLOT_PENICHON_GHAZALI_AIYAD_MENANI.ipynb` with 2 models `modele_NER_BERT` and `modele_sentiment_BERT` 
- the FastAPI `main.py`
- the main Database example `reviews_google.xlsx`

### Main notebook
- `Rapport_GEFFLOT_PENICHON_GHAZALI_AIYAD_MENANI.ipynb`: This is the main notebook containing every model and the explanations for each step. It serves as a report and does not need to be run.
- `Production_GEFFLOT_PENICHON_GHAZALI_AIYAD_MENANI.ipynb`: This is the main notebook containing the main model who will be used. It is ready to be run.

### FastAPI
- `main.py`: This is the FastAPI. It is ready to run.


*********************************************************************************************************************************************************
The `Annexe` file contain : 
## Data
- `reviews_google.xlsx`: This is the dataset of reviews that we webscrapped from the Tripadvisor website.
- `reviews_cleaned.xlsx`: This is the dataset of reviews after we cleaned the text.
- `reviews_target.xlsx`: This is the dataset where we tried to create a target by labeling the data with a BERT model.
- `reviews_absa.xlsx`: This is the dataset with a column containing the results of the ABSA model.
- `resultat_absa_cleaned_reviews.xlsx`: This is the dataset with the results of the ABSA model using cleaned data.

The following notebook was used to label the data:
- `Annotation using a RoBERTa model.ipynb`: This is the notebook that we used to do the labeling of the data.

## Files

The repository contains the following files:

### Specific models
- `Annotation using a RoBERTa model.ipynb`: This is the notebook used to label the data using a BERT model for train purposes.
- `NER subject and nouns.ipynb`: This notebook contains a specific model where we tried to do a named entity recognition (NER) task followed by a sentiment analysis task to deduce the positive and negative points of a certain hotel.
- `model-absa.ipynb`: This is a model where we explored the absa library, which does both a NER and sentiment analysis task.
- `model_ABSA_V1.ipynb`: This is an older version of `model-absa.ipynb`.
- `LSTM train .ipynb`: This contains the training of the LSTM model with word2vec.
- `LSTM test.ipynb`: This is the notebook containing the LSTM model.
- `sentiment-analysis-bert.ipynb`: This notebook deals with the sentiment analysis task using the BERT model.

### Statistic models
- `statistics.ipynb`: This notebook explores the output of the different models and showcases how the result would be used by hotel owners to get some insights on their performance.

### Supporting files
- `NER_vader_sentiment.csv`: This file contains the results for the NER and sentiment analysis model in `NER subject and nouns.ipynb`.
- `count_word fort faible.csv`: This file contains some overview of the results with some statistics.
- `function_clean_nlp.ipynb`: This file contains the function that helped us clean the text for the NLP project.
- `reviews_lstm.xlsx`: This is an Excel file containing the output of `LSTM test.ipynb`.
- `tokenizer.pickle`: This file contains the pickled version of the tokenizer for the LSTM model.

## Models

The following models were used in this project:
- BERT sentiment analysis model
- Vader sentiment analysis model
- Spacy NER model
- LSTM model for sentiment analysis
- ABSA library (combining NER and sentiment analysis)

## Additional Information

To run the code, you will need to install the necessary libraries and dependencies. 
