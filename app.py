import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import pygal
from flask import Flask, render_template, request, session, url_for, Response
from werkzeug.utils import redirect
from sklearn.model_selection import train_test_split
from sklearn import tree, neural_network
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from random import randint
from sklearn.neural_network import MLPClassifier
import time
import json
import sys, os, string
from sklearn import preprocessing
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


filepath = os.getcwd()
app = Flask(__name__)

global LR1,RF1,NB1,SVM1,clf, vectorizer,recallscore,precisionscore,accuracyscore,df

with open('classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def uploaddataset():
    return render_template('uploaddataset.html')


@app.route('/main')
def main():
    return render_template('main.html')

def ff(x_train,x_test, y_train, y_test):
    global X_trains,X_tests,y_trains,y_tests
    X_trains = pd.DataFrame(x_train)
    X_tests = pd.DataFrame(x_test)
    y_trains = pd.DataFrame(y_train)
    y_tests = pd.DataFrame(y_test)

@app.route('/train')
def traintestvalue():
    print("hello")
    return render_template('train.html')



@app.route('/traindataset', methods=['GET', 'POST'])
def traindataset():
    if request.method == "POST":
        session_var_value = session.get('filepath')
        df = pd.read_csv(session_var_value)
        value = request.form['traintestvalue']
        value1=(value)
        global vectorizer
        vectorizer = CountVectorizer()
        corpus = [
            'this is a sentence',
            'this is another sentence',
            'this is yet another sentence',
        ]
        X = vectorizer.fit_transform(corpus)
        # Preprocess the tweets
        df['processed_tweet'] = df['tweet_text'].apply(preprocess_tweet)
        df['processed_tweet'] = df['processed_tweet'].fillna('')

        # Remove rows with missing values
        df.dropna(inplace=True)

        # Extract features from the preprocessed tweets using TfidfVectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['processed_tweet'])

        # Split the data into training and testing sets
        y = df['cyberbullying_type']
        global X_train, X_test, y_train, y_test 
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=float(value1))
        ff(X_train,X_test, y_train, y_test)
        X_train1 = pd.DataFrame(X_train)
        X_trainlen=X_train.shape[0]
        y_test1 = pd.DataFrame(y_test)
        y_testlen =y_test.shape[0]
            # Train a classifier using LinearSVC
        clf = LinearSVC()
        clf.fit(X_train, y_train)
        

    # Save the trained classifier as a .pkl file
        with open('classifier.pkl', 'wb') as f:
            pickle.dump(clf, f)

    # save the trained vectorizer to a pickle file
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        return render_template('train.html',msg='sucess',data=X_train1.to_html(),X_trainlenvalue=X_trainlen,y_testlenval=y_testlen)
    return render_template('train.html')



def preprocess_tweet(tweet):
    # Check if tweet is a string
    if isinstance(tweet, str):
        # Convert to lowercase
        tweet = tweet.lower()
        # Remove URLs
        tweet = re.sub(r'http\S+', '', tweet)
        # Remove usernames
        tweet = re.sub(r'@\S+', '', tweet)
        # Remove hashtags
        tweet = re.sub(r'#\S+', '', tweet)
        # Remove punctuation
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        tweet = re.sub(r'\d+', '', tweet)
        # Remove extra whitespace
        tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        tweet = request.form['tweet']
        if tweet:
            # Preprocess the tweet
            processed_tweet = preprocess_tweet(tweet)
            # Vectorize the tweet
            vectorized_tweet = vectorizer.transform([processed_tweet])
            # Make the prediction
            prediction = clf.predict(vectorized_tweet)[0]
            # Return the prediction
            return render_template('prediction.html', prediction=prediction)
    return render_template('prediction.html')



@app.route('/uploaddataset',methods=["POST","GET"])
def uploaddataset_csv_submitted():
    if request.method == "POST":
        csvfile = request.files['csvfile']
        result = csvfile.filename
        filepath.replace("\\","/")
        file = filepath +"\\" + result
        print(file)
        session['filepath'] = file
        return render_template('uploaddataset.html',msg='sucess')
    return render_template('uploaddataset.html')

@app.route('/viewdata',methods=["POST","GET"])
def viewdata():
    session_var_value = session.get('filepath')
    print("session variable is=====" + session_var_value)
    df = pd.read_csv(session_var_value)
    global x,y
    y=df
    x = pd.DataFrame(df)
    x=x.dropna(how="any",axis=0)
    return render_template("viewdataset.html",col=x.columns.values, row_data=list(x.values.tolist()),zip=zip)



if __name__ == '__main__':
    app.secret_key = ".."
    app.run(debug=True)

