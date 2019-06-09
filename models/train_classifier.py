# import libraries
import pickle
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sqlite3
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from sklearn.metrics import classification_report, accuracy_score
import sys
import time


###############
#load the data#
###############

def load_data(database_filepath='data/DisasterResponse.db'):
    """
    Input:
    filepath of a sql database
    """
    
    #create engine
    
    engine = create_engine('sqlite:///' + database_filepath)


    
    #read sql assuming table name
    df = pd.read_sql('DisasterMessages', con= engine)
    
    #select messages as feature space
    X= df['message']
    #select targets, the genre column is a string and omitted
    Y= df.iloc[:, 5:]
    #get target categories
    categories = Y.columns.tolist()
    
    return X, Y, categories


###########
#tokenizer#
###########


def tokenize(text):
    #detect urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    #detect all urls
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        #replace all url with a placeholder
        text= text.replace(url, 'urlplaceholder')
        
    #normalize the text
    text = re.sub(r"[^a-zA-Z0-0]", ' ', text.lower())
        
    #tokenize text and assume english
    tokens = word_tokenize(text, language='english')
    
    # Remove Stopwords
    tokens = [x for x in tokens if x not in stopwords.words('english')]
    
    #remove words shorter than 2 characers
    tokens = [x for x in tokens if len(x)>2]
    
    #instantaite lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #use list comprehemsion to save on memory
    lemmed = [lemmatizer.lemmatize(tok, pos = 'n').strip() for tok in tokens]
    lemmed = [lemmatizer.lemmatize(tok, pos = 'v').strip() for tok in lemmed]
    
    return lemmed



######################
#build pipeline model#
######################


def make_model():
    
    """
    Inputs:
    - None
    Output:
    - Multioutput RF classifier gridsearch object
    
    """
    
    #instantiate a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    
    ])

    #set up paramaters for gridh search
    #uncomment cell blocks below if not in a rush

    parameters = {
        'vect__ngram_range': [(1,1), (1,2)],
        #'vect__max_df': [0.5, 0.75, 1.0],
        #'vect__max_features': [None, 20, 50],
        'tfidf__use_idf': [True, False],
        'clf__estimator__min_samples_split': [2, 3, 4]
                 }
    cv = GridSearchCV(pipeline, parameters)
    
    return cv

##################
#model evaluation#
##################

def model_evaluate(model, X_test, Y_test, categories):
    """
    Prints out precision, recall, and f1-score for trained model
    Inputs:
    - model: a trained model
    - X_test: the test features
    - Y_test: the test targets
    - categories: list of categories
    """
    
    Y_pred = model.predict(X_test)
    # Calculate the accuracy for each category
    for i in range(len(categories)):
        cat = categories[i]
        report = classification_report(Y_test.iloc[:, i].values, Y_pred[:,i])
        score = accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])
        print("Category: \t {}".format(cat))
        print(report)
        print("Accuracy: \t {:.2f}\n".format(score))
        
################
#save the model#
################


def save_model(model, filepath):
    '''
    Dump model as pickel
    Input: 
    - model: Model to be saved
    - filepath: path to model 
    '''
    pickle.dump(model, open(filepath, "wb"))
    print("model saved as {}".format(filepath))
    

################
#load the model#
################   


def load_model(path):
    with open(path, 'rb') as file:
        model_loaded = pickle.load(file)
    return model_loaded



###############
#main function#
###############


def main():

    #reads in the arguments 
    database_path, model_path = sys.argv[1:]
    
    #load the data
    X, Y, categories = load_data(database_path)
    
    #split data in train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    
    
    
    #create intance of model
    model = make_model()
    
    #record start
    start = time.time()
    #fit to training data
    model.fit(X_train, Y_train)

    delta = (time.time() - start)/60.0
    print("Model trained in {:.2f} minutes".format(delta))
    
    model_evaluate(model, X_test, Y_test, categories)
    
    save_model(model, path)
    
if __name__ == '__main__':
    main()
    