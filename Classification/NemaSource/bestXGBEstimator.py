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
import sys
from xgboost import plot_importance
from sklearn.metrics import log_loss

dataSize = int(sys.argv[1])
reconstuct = sys.argv[2]
modelName = "XGBSmeared" + str(dataSize)
mkdir_p(modelName)
# Load and transform data into sets 
# directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/3000s/'
# directory = '/mnt/opt/groups/jpet/NEMA_Image_Quality/3000s/'
directory = '/mnt/home/jbielecki1/MultiphotonClassification/NEMA/'
fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00'
df, X_train, X_test, y_train, y_test = createLearningBatches(directory + fileName, dataSize)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

bestXGB = pickle.load(open('XGB10e7/bestXGB.dat', 'rb'))
bestXGB.set_params(**{'n_estimators': 1500, 'max_depth': 7})
bestXGB.fit(X_train, y_train)
pickle.dump(bestXGB, open(modelName + "/trained" + modelName + ".dat", "wb"))

# make predictions for test data
y_pred_values = bestXGB.predict_proba(X_test)[:,1]
y_pred = (y_pred_values > 0.5)

# make predictions for train data
y_pred_values_train = bestXGB.predict_proba(X_train)[:,1]
y_pred_train = (y_pred_values_train > 0.5)

# Create predictions distributions
predictionsDistribution(y_test, y_pred_values, modelName = modelName, title = "IEC-" + modelName + "-Test")
predictionsDistribution(y_train, y_pred_values_train, modelName = modelName, title = "IEC-" + modelName + "-Train")

# Create ROC curves
createROC('XGB-train', y_train, y_pred_values_train, modelName = modelName)
createROC('XGB-test', y_test, y_pred_values, modelName = modelName)

# Create precision curves
drawPrecision(X_train, y_train, y_pred_values_train, modelName = modelName, title = 'XGB-train')
drawPrecision(X_test, y_test, y_pred_values,  modelName = modelName, title = 'XGB-test')

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
    reconstructionTest2D(FP, TP, modelName = modelName, title = 'IEC - XGB test recostrucion (TP + FP)')
    angleVsTime(FP, TP, TN, FN, modelName)

    plot_importance(bestXGB)
    plt.savefig(modelName + "/featureImportance.png")