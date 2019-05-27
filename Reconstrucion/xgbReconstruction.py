import pickle
import numpy as np
import pandas as pd

directory = '/mnt/home/jbielecki1/NEMA/'
modelFileName = '10000000/XGB/xgbEstimators1000Depth6'
max_depth = 6
feature_names = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'vol1', 'vol2', 'e1', 'e2', 'dt', 'rX1', 'rY1', 'rZ1', 'volD', 'lorL', 'deg3D', 'deg2D', 'rL', 'eSum']

model = pickle.load(open(directory + modelFileName, 'rb'))
data = pickle.load(open(directory + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part01', 'rb'))
data2 = pickle.load(open(directory + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part02', 'rb'))
data3 = pickle.load(open(directory + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part03', 'rb'))
allData = pd.concat([data, data2, data3])

codes = {1:1, 2:0, 3:0, 4:0}
y = allData["class"].map(codes)
x = allData.drop(["t1", "t2", "sX1", "sY1", "sZ1", "class", "rError"], axis = 1)
x.columns = ["f" + str(x) for x in range(20)]

y_pred_prob = model.predict_proba(x)
x.columns = feature_names
y_pred_prob = y_pred_prob
y_pred = y_pred_prob > 0.5

pPsPredictedPositive = x[y_pred[:,1]]
dataPositive = data.iloc[list(pPsPredictedPositive.index),:]
dataRec = dataPositive.iloc[:,:16]

dataRec.to_csv(directory + 'xgbReconstruction_parts3', sep = "\t", header = False, index = False)
