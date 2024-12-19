import sys
import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import pickle


def load_data(database_filepath):
    # Load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('message_cat', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    # Normalize text, tokenize, and remove stop words
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    return words


def build_model():
    # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Define parameter grid for GridSearchCV
    param_grid = {
        'clf__estimator__max_depth': [None, 10],
        'clf__estimator__min_samples_split': [2, 5]
    }

    # Set up GridSearchCV
    model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    # Predict on test set
    Y_pred = model.predict(X_test)

    # Evaluate each category
    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print("\n" + "="*60 + "\n")


def save_model(model, model_filepath):
    # Save model as a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
