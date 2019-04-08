#!/usr/bin/env python3.6

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sys
import dask.dataframe as dd
import numpy as np
import os

dataSize = int(sys.argv[2])
max_depth = int(sys.argv[1])
directory = '/mnt/home/jbielecki1/NEMA/' + str(dataSize) + "/"

def loadData():
    global X_train, X_test, y_train, y_test, class_test, class_train
    X_train = dd.from_pandas(pickle.load(open(directory + 'xTrain', 'rb')), npartitions = 10)
    X_test = dd.from_pandas(pickle.load(open(directory + 'xTest', 'rb')), npartitions = 10)
    y_train = dd.from_pandas(pickle.load(open(directory + 'yTrain', 'rb')), npartitions = 10)
    y_test = dd.from_pandas(pickle.load(open(directory + 'yTest', 'rb')), npartitions = 10)
    class_test = y_test[["class"]].to_dask_array()
    class_train = y_train[["class"]].to_dask_array()
    y_train = y_train[['newClass']].to_dask_array()
    y_test = y_test[['newClass']].to_dask_array()

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

modelName = "XGB"
loadData()
mkdir_p(directory + modelName)
n_estimators = 2000
modelFilePath = directory + modelName + "/xgbEstimators" + str(n_estimators) + "Depth" + str(max_depth)

if os.path.isfile(modelFilePath):
    model = pickle.load(open(modelFilePath + ".dat", 'rb'))
else:
    model = XGBClassifier(
        objective = 'binary:logistic', # Logistic regression for binary classification, output probability
        booster = 'gbtree', # Set estimator as gradient boosting tree
        subsample = 1, # Percentage of the training samples used to train (consider this)
        n_estimators = n_estimators, # Number of trees in each classifier
        learning_rate = 0.2, # Contribution of each estimator
        max_depth = max_depth, # Maximum depth of a tree
        colsample_bytree = 0.6, # The fraction of columns to be subsampled
    )
    results = {}
    eval_set = [( X_train, y_train), ( X_test, y_test)]
    model.fit(
        X_train, y_train, 
        early_stopping_rounds = 10, 
        eval_set=[(X_test, y_test)],
        eval_metric = ["error"],
        callbacks = [XGBClassifier.callback.record_evaluation(results)]
    )

train_accuracy = results['validation_0']['error']
test_accuracy = results['validation_1']['error']

# save model to file
pickle.dump(model, open(modelFilePath, "wb"), protocol=4)

bestAccuracy = max(test_accuracy)
bestNEstimators = test_accuracy.index(max(test_accuracy))
# Plot the results
plt.plot([i+1 for i in range(len(train_accuracy))], train_accuracy, label = "skuteczność - trening")
plt.plot([i+1 for i in range(len(test_accuracy))], test_accuracy, label = "skuteczność - test")
plt.xlabel("liczba drzew")
plt.ylabel("odsetek poprawnie sklasyfikowanych próbek")
plt.title("XGBoost accuracy (max_depth = " + str(max_depth) + ", best test accuracy: " + str(bestAccuracy) + ", n = " + str(bestNEstimators) + ")")
plt.legend(loc = "upper right")
plt.savefig(modelFilePath + ".png")
plt.clf()

