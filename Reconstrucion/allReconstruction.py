#!/usr/bin/env python3.6

import pickle
import numpy as np
import pandas as pd
import math

pathToFile = '/mnt/home/jbielecki1/NEMA/'
data1 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part01', 'rb'))
data2 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part02', 'rb'))
data3 = pickle.load(open(pathToFile + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part03', 'rb'))
data = pd.concat([data1, data2, data3])

cuttedData = data.apply(cutGeometry, axis = 1)
pPsPredictedPositive = pd.DataFrame(pd.concat([cuttedData[cuttedData == 1], cuttedData[cuttedData == 2]]).sort_index())
dataPositive = data.iloc[list(pPsPredictedPositive.index),:]
dataRec = dataPositive.iloc[:,:16]
dataRec.to_csv(pathToFile + 'allReconstruction', sep = "\t", header = False, index = False)