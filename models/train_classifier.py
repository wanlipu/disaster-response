import sys, re, pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multioutput import MultiOutputClassifier

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath, table_name='disaster'):
    """
    load data from database
    
    :param database_filepath: path for database file 
    :param table_name: name for the table, default is 'disaster'
    :return: X: pandas dataFrame for X
    :return: Y: labels 
    :return: category_names
    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df=pd.read_sql('SELECT * from {}'.format(table_name), engine)
    X = df['message']
    
    y = df.iloc[:,4:]
    category_names=y.columns.values
    
    return X, y, category_names


def tokenize(text):
    """
    tokenize input text
    
    :param text: input text string
    :return: clean_tokens: tokennized string list
    """
    
    # normalize text
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())

    # tokenize text
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]

    # stemming and lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    build machine learning pipeline
    
    :no input param
    :return: cv: Grid Search Model
    """
    # create pipeline
    pipeline = Pipeline([
        ('vect',TfidfVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(class_weight='balanced'),
                n_estimators=150)
            )
        )
    ])

    # set parameters for pipeline
    parameters = {
        'clf__estimator__base_estimator__min_samples_leaf': [2, 5],
        'clf__estimator__learning_rate': [0.1, 0.5]
    }

    # create grid search object
    # cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, scoring='f1_weighted')

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate model
    
    :param model: trained machine learning model
    :param X_test: test data
    :param Y_test: test labels
    :param category_names: category names for y
    :return: None
    """
    # generate predictions
    Y_pred = model.predict(X_test)

    # Print scores and save in log file
    with open('test.log','a+') as f:
        for i, name in enumerate(category_names):
            accu = accuracy_score(Y_test[:, i], Y_pred[:, i])
            prec = precision_score(Y_test[:, i], Y_pred[:, i], average='weighted')
            reca = recall_score(Y_test[:, i], Y_pred[:, i], average='weighted')
            f1 = f1_score(Y_test[:, i], Y_pred[:, i], average='weighted')
            score = "{}\n Accuracy: {:.4f}\t\t % Precision: {:.4f}\t\t % Recall: {:.4f}\t\t % F1_score: {:.4f}".format(
                name, accu, prec, reca, f1)
            print(score)
            f.write(score)


def save_model(model, model_filepath):
    """
    save model
    
    :param model: trained machine learning model
    :param model_filepath: path for model file
    :return: None
    """
    
    # save model
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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