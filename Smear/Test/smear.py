import pandas as pd
import matplotlib.pyplot as plt
import math

dataFrameNames = [
    "x1", # 1 gamma detected x position [mm]
    "y1", # 1 gamma detected y position [mm]
    "z1", # 1 gamma detected z position [mm]
    "x2", # 2 gamma detected x position [cm]
    "y2", # 2 gamma detected y position [cm]
    "z2", # 2 gamma detected z position [cm]
    "e1", # 1 gamma energy loss during detection [keV]
    "e2", # 2 gamma energy loss during detection [keV]
    "t1", # 1 gamma detection time [ps]
    "t2", # 2 gamma detection time [ps]
    "vol1", # 1 gamma volume name
    "vol2", # 2 gamma volume name
    "class", # Type of coincidence
]

def emissionPosition(row):
    sOfL = 300 # mm/ns
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
    return { 
        'x': (row['x1'] + row['x2'])/2 - dX,
        'y': (row['y1'] + row['y2'])/2 - dY,
        'z': (row['z1'] + row['z2'])/2 - dZ,
    }

directory = "/media/jasiek/F686ACAE86AC7133/Dokumenty/Studia/PracaMagisterska/ExampleData/"
filePath = directory + "pointsSourceExtractedData.csv"
filePathSmeard = directory + "pointsSourceExtractedDataSmeardCuted.csv"
df = pd.read_csv(filePath, names = dataFrameNames)
df["dt"] = df["t1"] - df["t2"]
dfSmeardCuted = pd.read_csv(filePathSmeard, names = dataFrameNames)
dfSmeardCuted["dt"] = dfSmeardCuted["t1"] - dfSmeardCuted["t2"]

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
plt.hist(df["z1"].transpose(), bins = 160, alpha = 0.7, range = [-270, 270])
plt.title("Współrzędna Z")
plt.xlabel("z [mm]")
plt.ylabel("Liczba fotonów")
plt.savefig("z.png", bbox_inches = "tight")

fig = plt.figure()
plt.hist(dfSmeardCuted["z1"].transpose(), bins = 160, alpha = 0.7, range = [-270, 270])
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
plt.hist(df["dt"].transpose(), bins = 160, alpha = 0.7, range = [-0.75, 0.75])
plt.title("Różnice czasów detekcji")
plt.xlabel("Różnica czasów detekcji [ns]")
plt.ylabel("Liczba par fotonów")
plt.savefig("time.png", bbox_inches = "tight")

fig = plt.figure()
plt.hist(dfSmeardCuted["dt"].transpose(), bins = 160, alpha = 0.7, range = [-0.75, 0.75])
plt.title("Różnice czasów (rozmycie i cięcie danych)")
plt.xlabel("Różnica czasów detekcji [ns]")
plt.ylabel("Liczba par fotonów")
plt.savefig("timeSmeardCutted.png", bbox_inches = "tight")

'''
emissionPoints = pd.DataFrame(columns = ['X', 'Y'])

for index, row in df.head(100000).iterrows():
    point = emissionPosition(row)
    emissionPoints = emissionPoints.append(
        {'X': point["x"],'Y': point["y"]},
        ignore_index = True
    )

fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111)
plt.hist2d(
    emissionPoints["X"], emissionPoints["Y"], bins = (100, 100), 
    cmap = plt.cm.jet, range = [[-15, 15], [-15, 15]]
)
plt.colorbar()
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
plt.title("Rekostrukcja źródła punktowego")
plt.savefig("reconstruction.png", bbox_inches = "tight")

emissionPointsSmeardCuted = pd.DataFrame(columns = ['X', 'Y'])

for index, row in dfSmeardCuted.head(100000).iterrows():
    point = emissionPosition(row)
    emissionPointsSmeardCuted = emissionPointsSmeardCuted.append(
        {'X': point["x"],'Y': point["y"]},
        ignore_index = True
    )

fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111)
plt.hist2d(
    emissionPointsSmeardCuted["X"], emissionPointsSmeardCuted["Y"], bins = (100, 100), 
    cmap = plt.cm.jet, range = [[-15, 15], [-15, 15]]
)
plt.colorbar()
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
plt.title("Rekostrukcja źródła punktowego (rozmycie i cięcie danych)")
plt.savefig("reconstructionSmeardCuted.png", bbox_inches = "tight")

'''