# Disaster Response Pipeline Project


## Summary

In this repository we aim to build a disaster response pipeline. The data consists of two datasets: tweets and their corresponding categories. The data is made available courtesy of Figure Eight Inc.

The end result is a web dashboard where users can insert a tweet and find out what category it falls under. As a bridge engineer in California, I am all too aware of an impending earthquake dubbed 'the big one' and the challenges that will come with it. Indeed in times of crisis, first responders and emergency services must be able to reach individuals quickly.

## Requirements

This package is written in python 3 and uses the following libraries:
- numpy
- pandas
- flask
- sklearn
- sqlalchemy
- nltk

You may also need to download these packages from nltk:
- punkt
- wordnet
- stopwords

## The Repository Files

### app

The web app built with Flask.

### data

The data is csv format as given, and the script for the ETL steps

### model

script to train and save a model. The saved model is bigger than 100MB, so not uploaded here.


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app, make sure you navigate the directory that contains the app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
