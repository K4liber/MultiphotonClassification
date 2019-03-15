#!/usr/bin/env python3.6
from calc import *

dataSize = int(sys.argv[1])
modelName = "STATS" + str(dataSize)
mkdir_p(modelName)
# directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/'
directory = '/mnt/opt/groups/jpet/NEMA_Image_Quality/3000s/'
fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00'
df, xTrain, xTest, yTrain, yTest = createLearningBatches(directory + fileName, 10000000)
xTest["class"] = yTest
pPs = xTest[xTest["class"] == 1]
reconstruction2D(pPs, title = "IEC reconstruction", modelName = modelName)
plotAngleVsTime(pPs, name = modelName, modelName = modelName)
