#!/usr/bin/env python3.6

import pickle
import numpy as np
import pandas as pd

pathToFile = '/mnt/home/jbielecki1/NEMA/'
modelFileName = '10000000/ADA/adaEstimators1000Depth6'
fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part'
dataPositiveParts = []
model = pickle.load(open(pathToFile + modelFileName, 'rb'))

def getGOJAFormatPositivePrediction(filePath):
    print("Processing file '" + filePath + "'.")
    data = pickle.load(open(filePath, 'rb'))
    x = data.drop(["t1", "t2", "sX1", "sY1", "sZ1", "class", "rError"], axis = 1)
    x.columns = ["f" + str(x) for x in range(20)]
    y_pred_prob = model.predict_proba(x)
    y_pred = y_pred_prob > 0.5
    pPsPredictedPositive = x[y_pred[:,1]]
    dataPositive = data.iloc[list(pPsPredictedPositive.index),:]
    return dataPositive.iloc[:,:16]

for i in range(8):
    dataPositiveParts.append(getGOJAFormatPositivePrediction(pathToFile + fileName + '0' + str(i+1)))

for i in range(9):
    if i != 3:
        dataPositiveParts.append(getGOJAFormatPositivePrediction(pathToFile + fileName + '1' + str(i)))
    
dataRec = pd.concat(dataPositiveParts)
dataRec.to_csv(pathToFile + 'adaReconstruction_parts16', sep = "\t", header = False, index = False)
