#!/usr/bin/env python3.6

import sys
projectDir = '/media/jasiek/F686ACAE86AC7133/Dokumenty/Studia/PracaMagisterska/MultiphotonClassification/'
if sys.argv[1] == "K":
    projectDir = '/mnt/home/jbielecki1/MultiphotonClassification/'
sys.path.append(projectDir + '/Classification/NemaSource')
from calc import dataFrameNames, reconstruction2D
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def loadData(nrows):
    global df, dfSmeardCuted
    dfSmeardCuted = pd.read_csv(
        '/mnt/home/jbielecki1/NEMA/' + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_partSMEAERED00', 
        sep = "\t", names = dataFrameNames(), nrows = nrows
    )
    df = pd.read_csv(
        directory + 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00', 
        sep = "\t", names = dataFrameNames(), nrows = nrows
    )
    df['dt'] = df['t1'] - df['t2']
    dfSmeardCuted['dt'] = dfSmeardCuted['t1'] - dfSmeardCuted['t2']

directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/3000s/'
dataSize = None

if sys.argv[1] == "K": directory = '/mnt/opt/groups/jpet/NEMA_Image_Quality/3000s/'
if len(sys.argv) > 2 and sys.argv[2]: dataSize = int(sys.argv[2])

loadData(dataSize) # energy threshold after smear = 200 keV

fig = plt.figure()
plt.hist(df["e1"].transpose(), bins = 160, alpha = 0.7, range = [20, 450])
plt.title("Widmo detekowanej energii")
plt.xlabel("Energia [keV]")
plt.ylabel("Liczba fotonów")
plt.savefig("energy.png", bbox_inches = "tight")

fig = plt.figure()
plt.hist(dfSmeardCuted["e1"].transpose(), bins = 160, alpha = 0.7, range = [20, 450])
plt.title("Widmo detekowanej energii (rozmycie i cięcie danych)")
plt.xlabel("Energia [keV]")
plt.ylabel("Liczba fotonów")
plt.savefig("energySmeardCutted.png", bbox_inches = "tight")

fig = plt.figure()
plt.hist(df["z1"].transpose(), bins = 160, alpha = 0.7, range = [-26, 26])
plt.title("Współrzędna Z")
plt.xlabel("z [mm]")
plt.ylabel("Liczba fotonów")
plt.savefig("z.png", bbox_inches = "tight")

fig = plt.figure()
plt.hist(dfSmeardCuted["z1"].transpose(), bins = 160, alpha = 0.7, range = [-26, 26])
plt.title("Współrzędna Z (rozmycie i cięcie danych)")
plt.xlabel("z [mm]")
plt.ylabel("Liczba fotonów")
plt.savefig("zSmeardCutted.png", bbox_inches = "tight")


fig = plt.figure()
plt.hist(df["x1"].transpose(), bins = 400, alpha = 0.7)
plt.title("Współrzędna X")
plt.xlabel("x [mm]")
plt.ylabel("Liczba fotonów")
plt.savefig("x.png", bbox_inches = "tight")

fig = plt.figure()
plt.hist(dfSmeardCuted["x1"].transpose(), bins = 400, alpha = 0.7)
plt.title("Współrzędna X (rozmycie i cięcie danych)")
plt.xlabel("x [mm]")
plt.ylabel("Liczba fotonów")
plt.savefig("xSmeardCutted.png", bbox_inches = "tight")

fig = plt.figure()
plt.hist(df["dt"].transpose(), bins = 160, alpha = 0.7, range = [-750, 750])
plt.title("Różnice czasów detekcji")
plt.xlabel("Różnica czasów detekcji [ns]")
plt.ylabel("Liczba par fotonów")
plt.savefig("time.png", bbox_inches = "tight")

fig = plt.figure()
plt.hist(dfSmeardCuted["dt"].transpose(), bins = 160, alpha = 0.7, range = [-750, 750])
plt.title("Różnice czasów (rozmycie i cięcie danych)")
plt.xlabel("Różnica czasów detekcji [ns]")
plt.ylabel("Liczba par fotonów")
plt.savefig("timeSmeardCutted.png", bbox_inches = "tight")

reconstruction2D(df, 'NEMA IEC - smeared data - reconstrucion', 'NEMA-SMEARED')
reconstruction2D(df, 'NEMA IEC - reconstrucion', 'NEMA-IDEAL')
