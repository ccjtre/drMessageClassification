import json
import plotly
import numpy as np
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
# from sklearn.externals import joblib
from joblib import load
from sqlalchemy import create_engine

# required libs for tokenizer
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# define regex pattern which we'll substitute in our tokenizer (all punctuation)
regex = re.compile('[%s]' % re.escape(string.punctuation))

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('categorized_messages', engine)
# print("tokenizing messages")
# df['tokenized_message'] = df['message'].apply(tokenize)
# print("gathering wordcounts")
# raw_word_counts = df['tokenized_message'].apply(pd.Series).stack().value_counts().head(n=10)

# load model
model = load("../models/categorize_message_final.joblib")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # metrics for genre percentage shares
    genre_value_counts = df['genre'].value_counts()
    genre_pcs = list(genre_value_counts.values)
    genre_names = list(genre_value_counts.index)
    
    # metrics for top ten category percentage shares - exclude 'related' category
    cat_columns = df.columns[np.logical_not(df.columns.isin(['id', 'message', 'original', 'genre']))]
    category_pcs = (100*(df[cat_columns[1:]].sum())/df.shape[0]).sort_values(ascending=False).head(10)
    category_pcs_values = list(category_pcs.values)
    category_pcs_names = list(category_pcs.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_pcs
                )
            ],

            'layout': {
                'title': 'Percentage share of message genres'
            }
        },
        {
            'data': [
                Bar(
                    x=category_pcs_names,
                    y=category_pcs_values
                )
            ],

            'layout': {
                'title': 'Top ten message categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()