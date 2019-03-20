#!/usr/bin/env python3.6

import pandas as pd
import numpy as np
import random as rnd
import sys
import math
import random
pathToDataLoad = '/media/jasiek/F686ACAE86AC7133/Dokumenty/Studia/PracaMagisterska/Nema_Image_Quality/3000s/'
pathToDataSave = pathToDataLoad

if sys.argv[1] == "K":
    pathToDataLoad = '/mnt/opt/groups/jpet/NEMA_Image_Quality/3000s/'
    pathToDataSave = '/mnt/home/jbielecki1/NEMA/'

fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_'

def dataFrameNames():
    return [
        "x1", # 1 gamma detected x position [cm]
        "y1", # 1 gamma detected y position [cm]
        "z1", # 1 gamma detected z position [cm]
        "t1", # 1 gamma detection time [ps]
        "x2", # 2 gamma detected x position [cm]
        "y2", # 2 gamma detected y position [cm]
        "z2", # 2 gamma detected z position [cm]
        "t2", # 2 gamma detection time [ps]
        "vol1", # 1 gamma volume ID
        "vol2", # 2 gamma volume ID
        "e1", # 1 gamma energy loss during detection [keV]
        "e2", # 2 gamma energy loss during detection [keV]
        "class", # Type of coincidence(1-true, 2-phantom-scattered, 3-detector-scattered, 4-accidental)
        "sX1", # 1 gamma emission x position [cm]
        "sY1", # 1 gamma emission y position [cm]
        "sZ1" # 1 gamma emission z position [cm]
    ] 

def emissionPoint(row):
    sOfL = 0.03 # cm/ps
    halfX = (row['x1'] - row['x2'])/2
    halfY = (row['y1'] - row['y2'])/2
    halfZ = (row['z1'] - row['z2'])/2
    LORHalfSize = math.sqrt(halfX**2 + halfY**2 + halfZ**2)
    versX = halfX/LORHalfSize
    versY = halfY/LORHalfSize
    versZ = halfZ/LORHalfSize
    dX = row['dt']*sOfL*versX/2
    dY = row['dt']*sOfL*versY/2
    dZ = row['dt']*sOfL*versZ/2

    return(  
        (row['x1'] + row['x2'])/2 - dX,
        (row['y1'] + row['y2'])/2 - dY,
        (row['z1'] + row['z2'])/2 - dZ,
    )

def distance(row):
    return math.sqrt(
            (row['sX1'] - row['RX1'])**2 + (row['sY1'] - row['RY1'])**2 + (row['sZ1'] - row['RZ1'])**2
        )

def reClass(row):
    rowClass = row['class']
    if (rowClass == 2 and row['emissionDistance'] < 0.05):
        rowClass = 1
    return rowClass

def shuffleTheOrder(row):
    rowCopy = row.copy()
    
    if bool(random.getrandbits(1)):
        rowCopy['x1'] = row['x2']
        rowCopy['y1'] = row['y2']
        rowCopy['z1'] = row['z2']
        rowCopy['e1'] = row['e2']
        rowCopy['t1'] = row['t2']
        rowCopy['vol1'] = row['vol2']
        rowCopy['x2'] = row['x1']
        rowCopy['y2'] = row['y1']
        rowCopy['z2'] = row['z1']
        rowCopy['e2'] = row['e1']
        rowCopy['t2'] = row['t1']
        rowCopy['vol2'] = row['vol1']
        rowCopy['dt'] = -1.0*row['dt']
        
    return rowCopy

for i in range(10):
    data = pd.read_csv(
        pathToDataLoad + fileName + "part0" + str(i), 
        sep = "\t", 
        names=dataFrameNames()
    )
    data['dt'] = data.apply (lambda row: row['t1'] - row['t2'], axis=1)
    data['emissionDistance'] = data.apply(lambda row:distance(row), axis=1)
    data['class'] = data.apply(lambda row:reClass(row), axis = 1)
    data = data.apply(lambda row:shuffleTheOrder(row), axis = 1)
    data = data.drop(['dt', 'RX1', 'RY1','RZ1', 'emissionDistance'], axis = 1)
    data.to_csv(
        pathToDataSave + fileName + 'REPAIRED_' + "part0" + str(i), 
        header=False, index=False, sep='\t'
    )
    print("Processed " + str(i+1) + "/19.")

for i in range(9):
    data = pd.read_csv(
        pathToDataLoad + fileName + "part1" + str(i), 
        sep = "\t", 
        names=dataFrameNames()
    )
    data['dt'] = data.apply (lambda row: row['t1'] - row['t2'], axis=1)
    data['emissionDistance'] = data.apply(lambda row:distance(row), axis=1)
    data['class'] = data.apply(lambda row:reClass(row), axis = 1)
    data = data.apply(lambda row:shuffleTheOrder(row), axis = 1)
    data = data.drop(['dt', 'RX1', 'RY1','RZ1', 'emissionDistance'], axis = 1)
    data.to_csv(
        pathToDataSave + fileName + 'REPAIRED_' + "part1" + str(i), 
        header=False, index=False, sep='\t'
    )
    print("Processed " + str(i+11) + "/19.")