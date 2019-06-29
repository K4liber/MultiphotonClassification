#!/usr/bin/env python3.6

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sys
import dask.dataframe as dd
import numpy as np
import os

dataSize = int(sys.argv[1])
max_depth = int(sys.argv[2])
n_estimators = int(sys.argv[3])
directory = '/mnt/home/jbielecki1/NEMA/' + str(dataSize) + "/"
attributes = [
    'e1',
    'e2',
    'eSum',
    'deg2D',
    'deg3D',
    'lorL',
    'dt'
]

def loadReducedData():
    global X_train, X_test, y_train, y_test, class_test, class_train
    X_train = dd.from_pandas(pickle.load(open(directory + 'xTrain', 'rb')), npartitions = 10)
    X_test = dd.from_pandas(pickle.load(open(directory + 'xTest', 'rb')), npartitions = 10)

    X_train = X_train[attributes]
    X_test = X_test[attributes]

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

modelName = "ADA"
loadReducedData(attributes)
mkdir_p(directory + modelName)
modelFilePath = directory + modelName + "/adaEstimatorsReduced" + str(n_estimators) + "Depth" + str(max_depth)

if os.path.isfile(modelFilePath):
    model = pickle.load(open(modelFilePath + ".dat", 'rb'))
else:
    model = AdaBoostClassifier(
        base_estimator = DecisionTreeClassifier(max_depth = max_depth),
        n_estimators = n_estimators,
        learning_rate = 0.2
    )
    model.fit(X_train, y_train)

test_accuracy = []
train_accuracy = []

for test_predicts, train_predicts in zip(model.staged_predict(X_test), model.staged_predict(X_train)):
    test_accuracy.append(accuracy_score(test_predicts, np.array(y_test)))
    train_accuracy.append(accuracy_score(train_predicts, np.array(y_train)))

# save model to file
pickle.dump(model, open(modelFilePath, "wb"), protocol=4)

bestAccuracy = max(test_accuracy)
bestNEstimators = test_accuracy.index(max(test_accuracy))
# Plot the results
plt.plot([i+1 for i in range(len(train_accuracy))], train_accuracy, label = "skuteczność - trening")
plt.plot([i+1 for i in range(len(test_accuracy))], test_accuracy, label = "skuteczność - test")
plt.xlabel("liczba drzew")
plt.ylabel("odsetek poprawnie sklasyfikowanych próbek")
plt.title("AdaBoost accuracy (max_depth = " + str(max_depth) + ", best test accuracy: " + str(bestAccuracy) + ", n = " + str(bestNEstimators) + ")")
plt.legend(loc = "upper right")
plt.savefig(modelFilePath + ".png")
plt.clf()
