#!/usr/bin/env python3.6

import pickle
import pandas as pd

directory = '/mnt/home/jbielecki1/NEMA/10000000/'
modelFileName = 'ADA/adaEstimators1000Depth4'
max_depth = 4

model = pickle.load(open(directory + modelFileName, 'rb'))
X_test = pickle.load(open(directory + 'xTest', 'rb'))
y_test = pickle.load(open(directory + 'yTest', 'rb'))
class_test = y_test[["class"]].values
y_test = y_test[['newClass']].values
y_pred_prob = model.predict_proba(X_test)[:,1]
y_pred = y_pred_prob > 0.5
bestNEstimators = 1000

def groupPerThreshold(X_test, y_test, y_pred_prob, modelName, resolution = 100):
    X_test = X_test
    points = pd.DataFrame(columns = ["Threshold", "FP", "TP", "TN", "FN"])
    pPsOrginalPositive = X_test[y_test > 0]
    pPsOrginalNegative = X_test[y_test == 0]
    minProb = min(y_pred_prob)
    maxProb = max(y_pred_prob)
    
    for i in range(resolution + 1):
        threshold = minProb + (maxProb-minProb)*float(i)/float(resolution)
        y_pred = y_pred_prob > threshold
        pPsPredictedPositive = X_test[y_pred == 1]
        pPsPredictedNegative = X_test[y_pred == 0]

        points = points.append({
            "Threshold": threshold,
            "FP": len(pd.merge(pPsPredictedPositive,pPsOrginalNegative, how='inner')),
            "TP": len(pd.merge(pPsPredictedPositive,pPsOrginalPositive, how='inner')),
            "TN": len(pd.merge(pPsPredictedNegative,pPsOrginalNegative, how='inner')),
            "FN": len(pd.merge(pPsPredictedNegative,pPsOrginalPositive, how='inner')),
        }, ignore_index = True)
    
    points.to_csv(directory + modelName + 'GroupPerThreshold' + str(bestNEstimators) + "d" + str(max_depth), sep = "\t", header = False, index = False)
    return points

points100 = groupPerThreshold(X_test, y_test, y_pred_prob, modelName = 'ADA')