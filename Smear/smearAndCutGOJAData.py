#!/usr/bin/env python3.6

import sys
projectDir = '/media/jasiek/F686ACAE86AC7133/Dokumenty/Studia/PracaMagisterska/MultiphotonClassification/'
if sys.argv[1] == "K":
    projectDir = '/mnt/home/jbielecki1/MultiphotonClassification/'
sys.path.append(projectDir + '/Classification/NemaSource')
from calc import dataFrameNames
import numpy as np
import dask.dataframe as dd
import pandas as pd
import math

def loadAllData():
    df = dd.read_csv(
        directory + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part*', 
        sep = "\t", names = dataFrameNames()
    )

def sigmaEnergy(energy, coeff = 0.0444):
    return coeff / math.sqrt(energy) * energy

def smearEnergy(energy):
    return np.random.normal(energy, 1000. * sigmaEnergy((energy) * 1. / 1000.))

def smearTime(time):
    return np.random.normal(time, 0.15) # Sigma = 150ps

def smearZ(z):
    newZ = np.random.normal(z, 1) # Sigma = 1cm
    
    if newZ > 25: newZ = 25
    if newZ < -25: newZ = -25

    return newZ

def smearX(volID):
    angle = math.radians(360 * (volID - 1)/384) # max volume ID = 384
    return math.cos(angle) * 43.5 # diameter = 85 cm

def smearY(volID):
    angle = math.radians(360 * (volID - 1)/384) # max volume ID = 384
    return math.sin(angle) * 43.5 # diameter = 85 cm

def smearAndCut(df, energyThreshold):
    df.loc[:,'e1'] = df['e1'].apply(smearEnergy)
    energyCut = df['e1'] >= energyThreshold
    df = df[energyCut]
    df.loc[:,'e2'] = df['e2'].apply(smearEnergy)
    energyCut = df['e2'] >= energyThreshold
    df = df[energyCut]
    df.loc[:,'t2'] = df['t2'].apply(smearTime)
    df.loc[:,'t1'] = df['t1'].apply(smearTime)
    df.loc[:,'z2'] = df['z2'].apply(smearZ)
    df.loc[:,'z1'] = df['z1'].apply(smearZ)
    df.loc[:,'x1'] = df['vol1'].apply(smearX)
    df.loc[:,'x2'] = df['vol2'].apply(smearX)
    df.loc[:,'y1'] = df['vol1'].apply(smearY)
    df.loc[:,'y2'] = df['vol2'].apply(smearY)
    return df

def smearAndCutData(energyThreshold, nrows):
    fileName = directory + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part'

    if sys.argv[1] == "K": 
        fileName = '/mnt/home/jbielecki1/NEMA/NEMA_IQ_384str_N0_1000_COINCIDENCES_part'
        
    for i in range(10):
        df = pd.read_csv(fileName+str(0)+str(i), sep = "\t", names = dataFrameNames(), nrows = nrows)
        smearAndCut(df, energyThreshold).to_csv(
            fileName+'SMEAERED'+str(0)+str(i), 
            header=False, index=False,sep='\t'
        )

    for i in range(9):
        df = pd.read_csv(fileName+str(1)+str(i), sep = "\t", names = dataFrameNames(), nrows = nrows)
        smearAndCut(df, energyThreshold).to_csv(
            fileName+'SMEAERED'+str(1)+str(i), 
            header=False, index=False,sep='\t'
        )

global directory, df
directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/3000s/'
if sys.argv[1] == "K": directory = '/mnt/opt/groups/jpet/NEMA_Image_Quality/3000s/'
smearAndCutData(200, None) # energy threshold after smear = 200 keV