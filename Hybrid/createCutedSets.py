#!/usr/bin/env python3.6

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
        "sZ1",    # 1 gamma emission z position [cm]
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

def createLearningBatches(filePath):
    df = pickle.load(open(filePath, 'rb'))
    codes = {1:1, 2:0, 3:0, 4:0}
    df["newClass"] = df["class"].map(codes)
    x = df.drop(["t1", "t2", "sX1", "sY1", "sZ1", "class", "rError", "newClass"], axis = 1)
    y = df[["class", "newClass"]]
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, train_size = 0.8, random_state = 42, stratify = y)
    print("Batches created successfully!")
    return xTrain, xTest, yTrain, yTest

directory = '/mnt/home/jbielecki1/NEMA/'
fileName = 'cutData'
mkdir_p(directory + "cut_parts4")
xTrain, xTest, yTrain, yTest = createLearningBatches(directory + fileName)
pickle.dump(xTest, open(directory + "cut_parts4" + "/xTest", 'wb'), protocol=4)
pickle.dump(xTrain, open(directory + "cut_parts4" + "/xTrain", 'wb'), protocol=4)
pickle.dump(yTest, open(directory + "cut_parts4" + "/yTest", 'wb'), protocol=4)
pickle.dump(yTrain, open(directory + "cut_parts4" + "/yTrain", 'wb'), protocol=4)