#!/usr/bin/env python3.6

import dask.dataframe as dd
import pandas as pd
import sys
import pickle
from sklearn.model_selection import train_test_split

def dataFrameNames():
    return [
        "x1",     # 1 gamma detected x position [cm]
        "y1",     # 1 gamma detected y position [cm]
        "z1",     # 1 gamma detected z position [cm]
        "t1",     # 1 gamma detection time [ps]
        "x2",     # 2 gamma detected x position [cm]
        "y2",     # 2 gamma detected y position [cm]
        "z2",     # 2 gamma detected z position [cm]
        "t2",     # 2 gamma detection time [ps]
        "vol1",   # 1 gamma volume ID
        "vol2",   # 2 gamma volume ID
        "e1",     # 1 gamma energy loss during detection [keV]
        "e2",     # 2 gamma energy loss during detection [keV]
        "class",  # Type of coincidence(1-true, 2-phantom-scattered, 3-detector-scattered, 4-accidental)
        "sX1",    # 1 gamma emission x position [cm]
        "sY1",    # 1 gamma emission y position [cm]
        "sZ1"     # 1 gamma emission z position [cm]
        "dt",     # Detection times difference
        "rX1",    # Reconstruction point - X cord
        "rY1",    # Reconstruction point - Y cord
        "rZ1",    # Reconstruction point - Z cord
        "rError", # Difference beetwen source point and recontructed point
        "volD",   # Volumes indexes difference
        "lorL",   # LOR length
        "deg3D",  # Angle beetwen lines (in XYZ geometry) connecting detection points with the center of detector
        "deg2D",  # Angle beetwen lines (in XY geometry) connecting detection points with the center of detector
        "rL",     # Distance beetween reconstructed point and the center of detector
        "eSum"    # Sum of the detecions energies
    ] 

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def createLearningBatches(filePath, size):
    if size > 10000000:
        dFrames = pd.DataFrame(columns = dataFrameNames())

        for i in range(10):
            dfPart = pickle.load(open(filePath + '0' + str(i), 'rb'))
            dFrames = dFrames.append(dfPart)

        for i in range(9):
            dfPart = pickle.load(open(filePath + '1' + str(i), 'rb'))
            dFrames = dFrames.append(dfPart)
    else:
        df = pickle.load(open(filePath + '00', 'rb')).head(size)
    
    codes = {1:1, 2:0, 3:0, 4:0}
    df["newClass"] = df["class"].map(codes)
    x = df.drop(["t1", "t2", "sX1", "sY1", "sZ1", "class", "rError"], axis = 1)
    y = df[["class", "newClass"]]
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, train_size = 0.8, random_state = 42, stratify = y)
    print("Batches created successfully!")
    return xTrain, xTest, yTrain, yTest

dataSize = int(sys.argv[1])
directory = '/mnt/home/jbielecki1/NEMA/'
fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_PREPARED_part'
mkdir_p(directory + str(dataSize))
xTrain, xTest, yTrain, yTest = createLearningBatches(directory + fileName, dataSize)
pickle.dump(xTest, open(directory + str(dataSize) + "/xTest", 'wb'))
pickle.dump(xTrain, open(directory + str(dataSize) + "/xTrain", 'wb'))
pickle.dump(yTest, open(directory + str(dataSize) + "/yTest", 'wb'))
pickle.dump(yTrain, open(directory + str(dataSize) + "/yTrain", 'wb'))