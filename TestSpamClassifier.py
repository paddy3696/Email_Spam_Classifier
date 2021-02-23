#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:48:55 2020

@author: padmanabhan, arun
"""

import pickle
import numpy as np
import os
from SpamClassifier import readTestFolder, preprocess
from striprtf.striprtf import rtf_to_text

      
def classify(fname, vecname, testfoldername, outputfile):    
    # load saved model and vectorizer
    if not os.path.isfile(fname) or not os.path.isfile(vecname):
        print('Model and/or vectorizer not available. Please build the model by running SpamClassifier.py.' +
              ' Make sure that the dataset is available in the same folder as of the script.')
        return
        
    model = pickle.load(open(fname, 'rb'))
    vectorizer = pickle.load(open(vecname, 'rb'))
    
    # load test data
    testdata = []
    for file, mail in readTestFolder(testfoldername):
        testdata.append(rtf_to_text(mail))

    print('preprocessing test data...')
    test_pre = [preprocess(message) for message in testdata]
    print('extracting features...')
    test_feature = vectorizer.transform(test_pre)
    print('predicting...')
    test_predictions = model.predict(test_feature)
    np.savetxt(outputfile, test_predictions, fmt = '%d')    # save predictions to file
    print('predictions written to file %s' % outputfile)
    
if __name__ == '__main__':
    classify('svmmodel.sav', 'vectorizer.sav', 'test', 'output.txt')