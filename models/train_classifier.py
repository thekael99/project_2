import sys

import sqlalchemy

import numpy as np
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

import pickle


def load_data(database_filepath="data/DisasterResponse.db"):
    engine = sqlalchemy.create_engine('sqlite:///' + str(database_filepath))
#     print(sqlalchemy.inspect(engine).get_schema_names())
    df = pd.read_sql_table("disaster", engine)
    X = df["message"]
#     print(df.columns)
    Y = df.drop(columns=["id", "message", "original", "genre"])    # 36 label
    return X, Y


def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(token).lower().strip() for token in tokens]
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vecttext', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

#     parameters = [
#         {
#             "clf": [RandomForestClassifier()],
#             "clf__n_estimators": [5, 50, 100, 250],
#             "clf__max_depth": [5, 8, 10],
#             "clf__random_state": [42],
#         },
#     ]

#     rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)

#     cv = GridSearchCV(
#         pipeline,
#         parameters,
#         cv=rkf,
#         scoring=["f1_weighted", "f1_micro", "f1_samples"],
#         refit="f1_weighted",
#         n_jobs=-1,
#     )

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for ind, column in enumerate(category_names):
        print(column, classification_report(Y_test.values[:, ind], y_pred[:, ind]))


def save_model(model, model_filepath="./models/classifier.pkl"):
    pickle.dump(model, open(model_filepath, 'wb'))
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#         print(Y)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, Y.columns.values)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
