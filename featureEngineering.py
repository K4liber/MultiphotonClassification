#!/usr/bin/env python3.6

import pandas as pd
import numpy as np
import math
import pickle
import sys

pathToData = '/media/jasiek/F686ACAE86AC7133/Dokumenty/Studia/PracaMagisterska/Nema_Image_Quality/3000s/'
if sys.argv[2] == 'K': pathToData = '/mnt/home/jbielecki1/NEMA/'
part = sys.argv[1] # part form "00" to "18"
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

def featureEngineering(row):
    sOfL = 0.03 # cm/ps
    dt = row["t1"] - row["t2"]
    halfX = (row['x1'] - row['x2'])/2
    halfY = (row['y1'] - row['y2'])/2
    halfZ = (row['z1'] - row['z2'])/2
    LORHalfSize = math.sqrt(halfX**2 + halfY**2 + halfZ**2)
    versX = halfX/LORHalfSize
    versY = halfY/LORHalfSize
    versZ = halfZ/LORHalfSize
    dX = dt*sOfL*versX/2
    dY = dt*sOfL*versY/2
    dZ = dt*sOfL*versZ/2
    dt = row['t1'] - row['t2']
    rX1 = (row['x1'] + row['x2'])/2 - dX
    rY1 = (row['y1'] + row['y2'])/2 - dY
    rZ1 = (row['z1'] + row['z2'])/2 - dZ
    rError = math.sqrt((row['sX1'] - rX1)**2 + (row['sY1'] - rY1)**2 + (row['sZ1'] - rZ1)**2)
    volD = min([(row['vol1'] - row['vol2'])%384, (row['vol2'] - row['vol1'])%384]) # 384 scintillators
    lorL = 2 * LORHalfSize
    cos3D = (row['x1']*row['x2']+row['y1']*row['y2']+row['z1']*row['z2'])/ \
            (math.sqrt(row['x1']**2+row['y1']**2+row['z1']**2)*math.sqrt(row['x2']**2+row['y2']**2+row['z2']**2))
    if cos3D > 1.0 : cos3D = 1.0 
    if cos3D < -1.0 : cos3D = -1.0 
    deg3D = math.degrees(math.acos(cos3D))
    cos2D = (row['x1']*row['x2']+row['y1']*row['y2'])/ \
            (math.sqrt(row['x1']**2+row['y1']**2)*math.sqrt(row['x2']**2+row['y2']**2))
    if cos2D > 1.0 : cos2D = 1.0 
    if cos2D < -1.0 : cos2D = -1.0 
    deg2D = math.degrees(math.acos(cos2D))
    rL = math.sqrt(rX1**2 + rY1**2 + rZ1**2)
    eSum = row['e1'] + row['e2']
    
    return (
        dt,     # Detection times difference
        rX1,    # Reconstruction point - X cord
        rY1,    # Reconstruction point - Y cord
        rZ1,    # Reconstruction point - Z cord
        rError, # Difference beetwen source point and recontructed point
        volD,   # Volumes indexes difference
        lorL,   # LOR length
        deg3D,  # Angle beetwen lines (in XYZ geometry) connecting detection points with the center of detector
        deg2D,  # Angle beetwen lines (in XY geometry) connecting detection points with the center of detector
        rL,     # Distance beetween reconstructed point and the center of detector
        eSum    # Sum of the detecions energies
    )

data = pd.read_csv(pathToData + fileName + "REPAIRED_part" + part, sep = "\t", names=dataFrameNames())
data[['dt','rX1','rY1','rZ1','rError','volD','lorL','deg3D','deg2D','rL','eSum']] = data.apply(lambda row: pd.Series(featureEngineering(row)), axis=1)
pickle.dump(data, open(pathToData + fileName + 'PREPARED_part' + part, 'wb'))