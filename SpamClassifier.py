#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:17:32 2020

@author: padmanabhan, arun
"""
#%%
algo = 'svm'
iscount = False
import pandas as pd 
import numpy as np 
import sklearn 
import time
import os
import string
import re
import pickle

#%%
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))    #get stopwords
lemmatizer = WordNetLemmatizer()

def preprocess(message):    
    lemmaWords = []
    for word, tag in pos_tag(word_tokenize(message)):
        if word in stop_words:  # remove stopwords
            continue
        word = word.lower()     #convert to lowercase
        tag_lower = tag.lower()[0]
        if tag_lower in ['a', 'r', 'n', 'v']:
            lemmaWords.append(lemmatizer.lemmatize(word, tag_lower))
        else:
            lemmaWords.append(word)

    message_pre = " ".join(lemmaWords)      #lemmatized sentence
    message_pre = message_pre.translate(str.maketrans('', '', string.punctuation + '\n\t\r' )) #remove punctuations and whitespaces
    message_pre = re.sub(' +',' ', message_pre)   #remove multiple whitespaces
    message_pre = re.sub('[0-9]+','', message_pre)   #remove numbers
    return message_pre

def readFolder(path):
    for fileName in os.listdir(path):
        isEnronDataset = (path.find('enron') != -1)
        filePath = os.path.join(path, fileName)
        if os.path.isfile(filePath):
            headerSeen, lines = False, []
            f = open(filePath, encoding="latin-1")
            for line in f:
                if isEnronDataset or headerSeen or line.startswith('Subject: '):
                    lines.append(line)
                elif line == '\n':
                    headerSeen = True
            f.close()
            content = '\n'.join(lines)
            yield content

numbers = re.compile(r'(\d+)')
file_pattern = re.compile(r'email[\d]+\.txt')

def sortByNumber(filename):
    if file_pattern.match(filename):
        filename_parts = numbers.split(filename)
        return int(filename_parts[1])
    else:
        return -1
            
def readTestFolder(path):
    for fileName in sorted(os.listdir(path), key = sortByNumber):
        filePath = os.path.join(path, fileName)
        if file_pattern.match(fileName) and os.path.isfile(filePath):
            lines = []
            f = open(filePath, encoding="latin-1")
            for line in f:
                lines.append(line)
            f.close()
            content = '\n'.join(lines)
            yield fileName, content
        
def buildModel():
    #%% Load dataset
    folders = {'easy_ham' : 0, 'easy_ham1' : 0, 'easy_ham_2' : 0, 'easy_ham_3' : 0, 'hard_ham' : 0, 'spam' : 1,
                'spam_2': 1, 'spam_3': 1, 'spam_4': 1, 'spam_5': 1, 'enron1/ham':0, 'enron1/spam':1,
                'enron2/ham':0, 'enron2/spam':1, 'enron3/ham':0, 'enron3/spam':1, 'enron4/ham':0, 'enron4/spam':1
                }
    # read in data
    rows = []
    print('Loading data...')
    for folder, label in folders.items():
        for mail in readFolder(folder):
            rows.append({'mail' : mail, 'label': label})
    
    data = pd.DataFrame(rows)
    data = data.dropna()
    #%% Train Test Split
    from sklearn.model_selection import train_test_split
    x_train,x_valid,y_train,y_valid = train_test_split(data['mail'], data['label'], test_size=0.2)
    
    #%% Preprocessing
    print('Preprocessing...')
    x_train_pre = [preprocess(message) for message in x_train]
    x_valid_pre = [preprocess(message) for message in x_valid]
    
    #%% Feature extraction
    print('feature extraction...')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = None
    if iscount:
        print('Using count vectorizer')
        vectorizer = CountVectorizer(ngram_range=(1,2))
    else:   
        print('Using  TFIDF vectorizer')
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        
    vectorizer.fit(x_train_pre)
    
    pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))
    
    x_train_feature = vectorizer.transform(x_train_pre)
    x_valid_feature = vectorizer.transform(x_valid_pre)
    
    #%%Train model
    print('Training model')
    model = None
    if algo == 'svm':
        print('Using SVM algorithm')
        from sklearn import svm
        model = svm.SVC(kernel='sigmoid', class_weight= 'balanced')
        model.fit(x_train_feature, y_train)
        pickle.dump(model, open('svmmodel.sav', 'wb'))
        print('Validating model..')
        y_pred = model.predict(x_valid_feature)
        
        sklearn.metrics.plot_confusion_matrix(estimator = model, X=x_valid_feature, y_true=y_valid)
        report = sklearn.metrics.classification_report(y_true = y_valid, y_pred = y_pred)
        print(report)
    else:
        print('Using Naive Bayes algorithm')
        from sklearn import naive_bayes
        nv = naive_bayes.MultinomialNB()
        nv.fit(x_train_feature, y_train)
        pickle.dump(nv, open('nbmodel.sav', 'wb'))
        y_pred_1 = nv.predict(x_valid_feature) 
        
        sklearn.metrics.plot_confusion_matrix(estimator = nv, X=x_valid_feature, y_true=y_valid)
        report_1 = sklearn.metrics.classification_report(y_true = y_valid, y_pred = y_pred_1)
        print(report_1)
        model = nv
    #%% Test with samples
    from striprtf.striprtf import rtf_to_text
    print('Testing with samples..')
    testdata = []
    testindex = []
    for name, testMail in readTestFolder('test'):
        testdata.append({'mail': rtf_to_text(testMail)})
        testindex.append(name)
    
    testdata = pd.DataFrame(testdata, index = testindex)
    
    test_pre = [preprocess(message) for message in testdata['mail']]
    test_feature = vectorizer.transform(test_pre)
    test_predictions = model.predict(test_feature)
    predictions = pd.DataFrame(test_predictions, index = testindex)
    
    print(predictions)
    
    #%% Test with external dataset
    from sklearn.metrics import confusion_matrix
    print('Testing with external dataset')
    testfolders = {'enron5/ham':0, 'enron5/spam':1, 'enron6/ham':0, 'enron6/spam':1 }
    # read in data
    rows = []
    print('Loading test data...')
    for folder, label in testfolders.items():
        for mail in readFolder(folder):
            rows.append({'mail' : mail, 'label': label})
    
    testdata = pd.DataFrame(rows)
    testdata = pd.concat([testdata, pd.read_csv('spam_or_not_spam.csv')])
    testdata = testdata.dropna()
    print('preprocessing test data...')
    test_pre = [preprocess(message) for message in testdata['mail']]
    print('extracting features...')
    test_feature = vectorizer.transform(test_pre)
    print('predicting...')
    test_predictions = model.predict(test_feature)
    predictions = pd.DataFrame(test_predictions)
    
    sklearn.metrics.plot_confusion_matrix(estimator = model, X=test_feature, y_true=testdata['label'])
    report = sklearn.metrics.classification_report(y_true = testdata['label'], y_pred = predictions)
    print(report)
    print(confusion_matrix(y_true = list(testdata['label']), y_pred = test_predictions))
    
#%%
if __name__ == '__main__':
    st = time.time()
    buildModel()
    et = time.time()
    tt = (et-st)/60
    print('Time taken : %.2f mins ' % (tt))