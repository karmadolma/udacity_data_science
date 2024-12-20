# Disaster Response Pipeline Project

## Project Summary
The Disaster Response Pipeline Project is a data engineering and machine learning pipeline designed to classify disaster-related messages. The goal is to help organizations like NGOs and disaster response teams prioritize and route messages to the appropriate department during emergencies.

### The pipeline:

1. Cleans and processes raw message data.
2. Trains and evaluates a machine learning model for classifying messages into multiple categories.
3. Deploys a web app where users can input new messages for real-time classification.

## File Structure
The repository contains the following files and directories:

data/
│   process_data.py         # ETL pipeline script
│   disaster_messages.csv   # Dataset with disaster messages
│   disaster_categories.csv # Dataset with message categories
│   DisasterResponse.db     # SQLite database with cleaned data
models/
│   train_classifier.py     # ML pipeline script
│   classifier.pkl          # Saved ML model
app/
│   run.py                  # Flask web app script
│   templates/              # HTML templates for the web app
README.md                   # Project documentation

## How to Run the Project
1. Setting Up the Environment
Ensure you have the following Python libraries installed:

- pandas
- numpy
- nltk
- scikit-learn
- SQLAlchemy
- Flask
- joblib

You can install these dependencies using:

pip install -r requirements.txt

2. Running the ETL Pipeline
Run the ETL pipeline to process and clean the data, and store it in a database. Use the following command from the root directory:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

3. Training the ML Model
Run the machine learning pipeline to train the classifier and save the model. Use the following command:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

4. Running the Web Application
Navigate to the app/ directory and start the Flask web app:

python run.py

The app will run locally on your machine. Open your browser and go to: http://0.0.0.0:3001/ or you will find the link in your command prompt/terminal.

5. Using the Web Application
1. Input a disaster-related message in the text box.
2. Click the "Classify Message" button to see the predicted categories of the message.

### Code Documentation
The codebase includes:

Comments: Inline comments explaining complex or key parts of the code.
Docstrings: Each function has a descriptive docstring that explains its purpose, inputs, and outputs.

Example of a function docstring:

def tokenize(text):
    """
    Tokenizes and lemmatizes text data.

    Args:
        text (str): The text to be processed.

    Returns:
        list: A list of cleaned and lemmatized tokens.
    """

### Acknowledgments
This project is part of the Udacity Data Science Nanodegree Program. The datasets used in this project were provided by Figure Eight. Special thanks to the program mentors and peers for their guidance.













