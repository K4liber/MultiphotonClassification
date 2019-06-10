#!/usr/bin/env python3.6

import pickle
import numpy as np
import pandas as pd

pathToFile = '/mnt/home/jbielecki1/NEMA/'
modelFileName = '10000000/ADA/adaEstimators1000Depth6'

model = pickle.load(open(pathToFile + modelFileName, 'rb'))
data1 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part01', 'rb'))
data2 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part02', 'rb'))
data3 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part03', 'rb'))
data4 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part04', 'rb'))
data5 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part05', 'rb'))
data6 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part06', 'rb'))
data7 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part07', 'rb'))
data8 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part08', 'rb'))
data10 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part10', 'rb'))
data11 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part11', 'rb'))
data12 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part12', 'rb'))
data13 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part13', 'rb'))
data14 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part14', 'rb'))
data15 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part15', 'rb'))
data16 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part16', 'rb'))
data17 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part17', 'rb'))

data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data10, data11, data12, data13, data14, data15, data16, data17], ignore_index = True)

codes = {1:1, 2:0, 3:0, 4:0}
y = data["class"].map(codes)
x = data.drop(["t1", "t2", "sX1", "sY1", "sZ1", "class", "rError"], axis = 1)

y_pred_prob = model.predict_proba(x)
y_pred = y_pred_prob > 0.5

pPsPredictedPositive = x[y_pred]
dataPositive = data.iloc[list(pPsPredictedPositive.index),:]
dataRec = dataPositive.iloc[:,:16]

dataRec.to_csv(directory + 'adaReconstruction_parts16', sep = "\t", header = False, index = False)
