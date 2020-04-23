# Disaster Response Pipeline Project

This is part of Udacity Data Scientist Program, and this repository is meant to document my progress towards `Disaster Response Pipeline Project` completion.

### Project Summary

In the Project, I worked on a data set containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

This project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


### Repo directory structure

    ├── README.md
    ├── models
    |   ├── train_classifier.py
    |   └── classifier.pkl          # saved model 
    ├── data
    |   ├── process_data.py
    |   ├── disaster_categories.csv # data to process 
    |   ├── disaster_messages.csv   # data to process
    |   └── DisasterResponse.db     # database to save clean data to
    ├── app
        ├── run.py
        └── templates
            ├── go.html             # classification result page of web app
            └── master.html         # main page of web app

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/
