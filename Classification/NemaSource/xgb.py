#!/usr/bin/env python3.6

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from calc import *
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn.metrics import confusion_matrix
import pickle

dataSize = int(sys.argv[1])
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
modelFilePath = directory + modelName + "/xgbEstimatorsCV" + str(n_estimators)

# fit model on training data
model = XGBClassifier(
    objective = 'binary:logistic', # Logistic regression for binary classification, output probability
    booster = 'gbtree', # Set estimator as gradient boosting tree
    subsample = 1, # Percentage of the training samples used to train (consider this)
    n_estimators = n_estimators # Number of trees in each classifier
)

param_dist = {
    'learning_rate': stats.uniform(0.25, 0.1), # Contribution of each estimator
    'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], # Maximum depth of a tree
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1], # The fraction of columns to be subsampled
    'min_child_weight': [1, 2, 3, 4]    # Minimum sum of instance weight (hessian) needed in a child 8
                                        # In linear regression task, this simply corresponds to minimum 
                                        # number of instances needed to be in each node
}

clf = RandomizedSearchCV(
    model,
    param_distributions = param_dist,  
    n_iter = 20, 
    cv = 3, # Cross-validation number of folds
    scoring = 'roc_auc',
    eval_set = [(X_test, y_test)],
    early_stopping_rounds = 25,
    error_score = 0, 
    verbose = 2, 
    n_jobs = -1
)

clf.fit(X_train, y_train)

# make predictions for test data
y_pred_values = clf.predict(X_test)
y_pred = (y_pred_values > 0.5)

# make predictions for train data
y_pred_values_train = clf.predict(X_train)
y_pred_train = (y_pred_values_train > 0.5)

# Create ROC curves
createROC('XGB-train', y_train, y_pred_values_train, modelName = modelName)
createROC('XGB-test', y_test, y_pred_values, modelName = modelName)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (test): %.2f%%" % (accuracy * 100.0))
accuracyTrain = accuracy_score(y_train, y_pred_train)
print("Accuracy (train): %.2f%%" % (accuracyTrain * 100.0))

# Creating the Confusion Matrixes
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(
    cm, 
    classes=['not pPs', 'pPs'],
    title='XGB-test',
    accuracy='Accuracy: ' + '%.2f' % (accuracy * 100.0) + '%, size: ' + str(y_pred.size),
    modelName = modelName
)
cmTrain = confusion_matrix(y_train, y_pred_train)
plot_confusion_matrix(
    cmTrain, 
    classes=['not pPs', 'pPs'],
    title = 'XGB-train',
    accuracy = 'Accuracy: ' + '%.2f' % (accuracyTrain * 100.0) + '%, size: ' + str(y_pred_train.size),
    modelName = modelName
)

# save best model and all results to file
pickle.dump(clf.best_estimator_, open(modelName + "/bestXGB.dat", "wb"))
pickle.dump(clf.cv_results_, open(modelName + "/CVresults.dat", "wb"))
# plot single tree
# fig = plt.figure()
# fig.set_size_inches(3600, 2400)
# ax = plot_tree(clf.best_estimator_, rankdir='LR')
# plt.tight_layout()
# plt.savefig(modelName + "/bestTree.png", dpi = 600)