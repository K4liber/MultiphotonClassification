#!/usr/bin/env python3.6

import sys
projectDir = '/media/jasiek/F686ACAE86AC7133/Dokumenty/Studia/PracaMagisterska/MultiphotonClassification/'
if sys.argv[1] == "K":
    projectDir = '/mnt/home/jbielecki1/MultiphotonClassification/'

sys.path.append(projectDir + '/Classification/NemaSource')
from calc import dataFrameNames
import numpy as np
import dask.dataframe as dd

def loadData(dataSize):
    directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/3000s/'
    if sys.argv[1] == "K": directory = '/mnt/opt/groups/jpet/NEMA_Image_Quality/3000s/'
    global df
    df = dd.read_csv(
        directory + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part*', 
        sep = "\t", names = dataFrameNames()
    ).head(dataSize, npartitions=-1)

dataSize = int(sys.argv[2])
loadData(dataSize)
print(len(df["t1"]))