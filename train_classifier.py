import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import string
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

# define regex pattern which we'll substitute in our tokenizer (all punctuation)
regex = re.compile('[%s]' % re.escape(string.punctuation))

def load_data(database_filepath):
    """ loads 'categorized_messages' table from sqlite db
    and extracts feature and target array information for message categorization
    Parameters:
        database_filepath (str): path th sqlite db, must have a 'categorized_messages' table defined
    Returns:
        X (array): feature array.
        Y (array): multi-output array
        cat_columns (array): human-readable names for Ys categories
    """
    df = pd.read_sql_table('categorized_messages', 'sqlite:///' + database_filepath)
    # df = df.head(n=200)
    X = df.message.values
    cat_columns = df.columns[np.logical_not(df.columns.isin(['id', 'message', 'original', 'genre']))]
    Y = df[cat_columns].values
    return X, Y, cat_columns

def tokenize(text):
    """ text tokenizer for CountVectoriser transformer
    removes punctuation, lowers casing, strips words of whitespace,
    lemmatizes and removes english stop words
    Parameters:
        text (str): raw text message
    Returns:
        clean_tokens (str list): cleaned token representation of text.
    """
    text = regex.sub(' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok.lower().strip())
        clean_tokens.append(clean_tok)
    clean_tokens = [tok for tok in clean_tokens if tok not in stopwords.words("english")]
    return clean_tokens

def build_model():
    """ builds GridSearch estimator pipeline with RandomForestClassifier as base,
    CountVectorizer and TfidfTransformer as transformers.
    tokenize function must be defined
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'vect__max_df': (0.5, 0.75, 1.0), 
              'tfidf__use_idf': (True, False)
             }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ prints precision and recall of model, a MultiOutputClassifier 
    Parameters:
        model (MultiOutputClassifier):
        X_test (array): test feature array
        Y_test (array): test target array (multi-output)
        category_names (array): human-readable names for Y_test categories
    Returns:
        None
    """
    Y_pred = model.predict(X_test)
    for cat, ci in zip(category_names, range(Y_test.shape[1])):
        y_test = Y_test[:, ci]
        y_pred = Y_pred[:, ci]
        print("classification report for: {}\n".format(cat))
        print(classification_report(y_test, y_pred))
    return None

def save_model(model, model_filepath):
    """saves model to file at model_filepath"""
    dump(model, model_filepath)
    return None

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