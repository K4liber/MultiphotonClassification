#!/usr/bin/env python3.6

import pickle
import numpy as np
import pandas as pd
import math

pathToFile = '/mnt/home/jbielecki1/NEMA/'
data1 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part01', 'rb'))
data2 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part02', 'rb'))
data3 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part03', 'rb'))
data = pd.concat([data, data2, data3])

# Cut params
width1 = 1050.0
height1 = 42.0
center = 180.0
width2 = 160.0
height2 = 6.5
zCut = 10.85

dataClass1 = data[data['class'] == 1]
dataClass2 = data[data['class'] == 2]
dataClass3 = data[data['class'] == 3]
dataClass4 = data[data['class'] == 4]

def ellipseY(x, width, height, center):
    return center - height*math.sqrt(1 - x**2/width**2)

xEllipse1 = np.arange(-width1, width1+1)
yEllipse1 = np.array([ ellipseY(el, width1, height1, center) for el in xEllipse1 ])
xEllipse2 = np.arange(-width2, width2+1)
yEllipse2 = np.array([ ellipseY(el, width2, height2, center) for el in xEllipse2 ])

def cutGeometry(row):
    prediction = True
    rowClass = row['class']
    
    # Check z
    if row['rZ1'] > zCut or row['rZ1'] < -zCut:
        prediction = False
        
    # Check ellipse1
    if row['dt'] < -width1 or row['dt'] > width1:
        prediction = False
    else:
        if row['deg2D'] < ellipseY(row['dt'], width1, height1, center):
            prediction = False
    
    # Check ellipse2
    if row['dt'] > -width2 and row['dt'] < width2 \
        and row['deg2D'] > ellipseY(row['dt'], width2, height2, center):
        prediction = False
    
    if prediction and row['class'] == 1:
        return 1 # TP
    elif prediction and row['class'] != 1:
        return 2 # FP
    elif ~prediction and row['class'] != 1:
        return 3 # TN
    elif ~prediction and row['class'] == 1:
        return 4 # FN

cuttedData = data.apply(cutGeometry, axis = 1)
pPsPredictedPositive = pd.DataFrame(pd.concat([cuttedData[cuttedData == 1], cuttedData[cuttedData == 2]]).sort_index())
dataPositive = data.iloc[list(pPsPredictedPositive.index),:]
dataRec = dataPositive.iloc[:,:16]
dataRec.to_csv(pathToFile + 'cutReconstruction', sep = "\t", header = False, index = False)