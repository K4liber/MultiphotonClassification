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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import pickle
import sys
from sklearn.metrics import log_loss

def loadData():
    directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/'
    # directory = '/mnt/opt/groups/jpet/NEMA_Image_Quality/3000s/'
    fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00'
    global df, X_train, X_test, y_train, y_test
    df, X_train, X_test, y_train, y_test = createLearningBatches(directory + fileName, 1000)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

dataSize = int(sys.argv[1])
n_estimators = int(sys.argv[2])
max_depth = int(sys.argv[3])
reconstuct = sys.argv[4]
modelName = "ADA" + str(dataSize)
mkdir_p(modelName)
loadData()

# fit model on training data
model = AdaBoostClassifier(
    base_estimator = DecisionTreeClassifier(max_depth = max_depth),
    algorithm = 'SAMME.R',
    n_estimators = n_estimators,
    learning_rate = 0.2
)

model.fit(X_train, y_train)

# make predictions for test data
y_pred_values = model.predict(X_test)
y_pred = (y_pred_values > 0.5)

# make predictions for train data
y_pred_values_train = model.predict(X_train)
y_pred_train = (y_pred_values_train > 0.5)

# Create ROC curves
createROC('ADA-train', y_train, y_pred_values_train, modelName = modelName)
createROC('ADA-test', y_test, y_pred_values, modelName = modelName)

# evaluate predictions
evaluateFile = open(modelName + '/evaluation.dat', 'w')
accuracy = accuracy_score(y_test, y_pred)
evaluateFile.write("Accuracy (test): %.2f%%\n" % (accuracy * 100.0))
accuracyTrain = accuracy_score(y_train, y_pred_train)
evaluateFile.write("Accuracy (train): %.2f%%\n" % (accuracyTrain * 100.0))
logLossTest = log_loss(y_test, y_pred_values)
evaluateFile.write("Log loss (test): %.2f\n" % (logLossTest))
logLossTrain = log_loss(y_train, y_pred_values_train)
evaluateFile.write("Log loss (train): %.2f\n" % (logLossTrain))

# Creating the Confusion Matrixes
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(
    cm, 
    classes=['not pPs', 'pPs'],
    title='ADA-test',
    accuracy='Accuracy: ' + '%.2f' % (accuracy * 100.0) + '%, size: ' + str(y_pred.size),
    modelName = modelName
)
cmTrain = confusion_matrix(y_train, y_pred_train)
plot_confusion_matrix(
    cmTrain, 
    classes=['not pPs', 'pPs'],
    title = 'ADA-train',
    accuracy = 'Accuracy: ' + '%.2f' % (accuracyTrain * 100.0) + '%, size: ' + str(y_pred_train.size),
    modelName = modelName
)

# save model
pickle.dump(model, open(modelName + "/ADAmodel.dat", "wb"))
# plot single tree
# fig = plt.figure()
# fig.set_size_inches(3600, 2400)
# ax = plot_tree(clf.best_estimator_, rankdir='LR')
# plt.tight_layout()
# plt.savefig(modelName + "/bestTree.png", dpi = 600)

if reconstuct == "T":
    pPsOrginalPositive = X_test[y_test > 0]
    pPsOrginalNegative = X_test[y_test == 0]
    pPsPredictedPositive = X_test[y_pred]
    pPsPredictedNegative = X_test[y_pred == 0]

    FP = pd.merge(pPsPredictedPositive,pPsOrginalNegative, how='inner')
    TP = pd.merge(pPsPredictedPositive,pPsOrginalPositive, how='inner')
    TN = pd.merge(pPsPredictedNegative,pPsOrginalNegative, how='inner')
    FN = pd.merge(pPsPredictedNegative,pPsOrginalPositive, how='inner')

    saveHistograms(FP, TP, TN, FN, modelName)
    reconstructionTest2D(FP, TP, modelName = modelName, title = 'IEC - ADA test recostrucion (TP + FP)')
    angleVsTime(FP, TP, TN, FN, modelName)
    plotFeatureImportances(X_train.columns, model.feature_importances_, modelName)
