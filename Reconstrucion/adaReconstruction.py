#!/usr/bin/env python3.6

import pickle
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

directory = '/mnt/home/jbielecki1/NEMA/'
modelFileName = '10000000/ADA/adaEstimators1000Depth6'

model = pickle.load(open(directory + modelFileName, 'rb'))
data = pickle.load(open(directory + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part01', 'rb'))
data2 = pickle.load(open(directory + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part02', 'rb'))
data3 = pickle.load(open(directory + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part03', 'rb'))
allData = pd.concat([data, data2, data3])

codes = {1:1, 2:0, 3:0, 4:0}
y = allData["class"].map(codes)
x = allData.drop(["t1", "t2", "sX1", "sY1", "sZ1", "class", "rError"], axis = 1)

test_accuracy = []
max_acc = 0
y_pred_prob = []
y_pred = []

for test_predicts_el in model.staged_predict_proba(x):
    y_pred_prob_el = test_predicts_el[:,1]
    y_pred_el = y_pred_prob_el > 0.5
    acc = accuracy_score(y_pred_el, np.array(y))
    
    if acc > max_acc:
        max_acc = acc
        y_pred_prob = y_pred_prob_el
        y_pred = y_pred_el
        
    test_accuracy.append(acc)

pPsPredictedPositive = x[y_pred]
dataPositive = data.iloc[list(pPsPredictedPositive.index),:]
dataRec = dataPositive.iloc[:,:16]

dataRec.to_csv(directory + 'adaReconstruction', sep = "\t", header = False, index = False)
