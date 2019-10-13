# drMessageClassification
Train a model to categorize text messages in a disaster response scenario.
The trained model is used in a web app to allow users to classify custom messages into various disaster-response-related categories.

# Usage:
This project is divided into three sequential modules: an ETL pipeline, an ML pipeline and a Flask web app.

### Running the ETL pipeline (data/process_data.py):
process_data.py takes three arguments: relative paths to CSV files for both the messages and their categorisations
(linked by a common id field), and the desired relative path for the output sqllite database file containing the merged
and cleaned data:

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

### Running the ML pipeline (models/train_classifier.py):
train_classifier.py takes two arguments: the relative path to the output sqllite database file of process_data.py, 
and the desired relative path for its output pickle file, which will contain a multi-output classification model trained on said database:

`python models/train_classifier.py data/DisasterResponse.db models/categorize_message_final.joblib`

### Running the Flask web app (app/run.py):
The web app loads the previously created database and classifier model to display visualisations of message metrics
and to allow users to classify new messages in an interactive environment.

In the app directory run:

`python run.py`

Obtain the URL details for the running web app by running the following in a seperate terminal:

`env | grep WORK`

You'll see an output resembling:

`WORKSPACEDOMAIN=udacity-student-workspaces.com`

`WORKSPACEID=viewabcdefg1`

The URL will be `https://SPACEID-3001.SPACEDOMAIN`
In this case, `https://viewabcdefg1-3001.udacity-student-workspaces.com`

Please note that both the ML pipeline and the webapp depend on the tablename in the database file being called "categorized_messages", 
and that the flask app depends on the database itself and the pickle file being called "DisasterResponse.db" and 
"categorize_message_final.joblib" respectively.

# Libraries:
sys, numpy, pandas, sqlalchemy, nltk, re, string, sklearn, joblib, json, plotly, flask

# File overview:
1. data/disaster_messages.csv: raw messages CSV. Id is a foreign key to categories.csv.
2. data/disaster_categories.csv: raw categories CSV.
3. data/process_data.py: ETL pipeline module, creates table of categorized messages in sqllite database.
4. data/DisasterResponse.db: Example output of data/process_data.py.
5. models/train_classifier.py: trains gridsearch-optimized multi-output classification model on the ETL pipeline output database.
6. app/run.py: web app script
7. app/templates/master.html: landing page HTML.
8. app/templates/go.html: message classification result HTML.
