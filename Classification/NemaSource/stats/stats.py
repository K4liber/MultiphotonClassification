import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math

dataFrameNames = [
    "x1", # 1 gamma detected x position
    "y1", # 1 gamma detected y position
    "z1", # 1 gamma detected z position
    "t1", # 1 gamma detection time
    "x2", # 2 gamma detected x position
    "y2", # 2 gamma detected y position
    "z2", # 2 gamma detected z position
    "t2", # 2 gamma detection time
    "vol1", # 1 gamma volume ID
    "vol2", # 2 gamma volume ID
    "e1", # 1 gamma energy loss during detection
    "e2", # 2 gamma energy loss during detection
    "class", # Type of coincidence(1-true, 2-phantom-scattered, 3-detector-scattered, 4-accidental)
    "sX1", # 1 gamma emission x position
    "sY1", # 1 gamma emission y position
    "sZ1" # 1 gamma emission z position
]

def createLearningBatches(fileName):
    df = pd.read_csv(fileName, sep = "\t", names = dataFrameNames)
    df["dt"] = df["t1"] - df["t2"]
    codes = {1:1, 2:0, 3:0, 4:0}
    df["class"] = df["class"].map(codes)
    x = df.drop(["t1", "t2", "sX1", "sY1", "sZ1", "class"], axis = 1)
    y = df[["class"]]
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)
    return df, xTrain, xTest, yTrain, yTest

def createStats(df, name):
    plt.figure()
    plt.hist(df[["e1"]].transpose(), bins=40, edgecolor='k', alpha=0.7)
    e1Mean = df["e1"].mean()
    plt.axvline(e1Mean, color='k', linestyle='dashed', linewidth=1)
    plt.title('Energy loss - ' + name)
    plt.xlabel('Energy [keV]')
    plt.ylabel('#')
    plt.savefig(name + 'Energy.png')

    plt.figure()
    plt.hist(df[["dt"]].transpose(), bins=20, edgecolor='k', alpha=0.7)
    plt.title('Detection time difference - ' + name)
    plt.xlabel('time difference [us]')
    plt.ylabel('#')
    plt.savefig(name + 'Time.png')

    plt.figure()
    plt.hist(df[["x1"]].transpose(), bins=20, edgecolor='k', alpha=0.7)
    plt.title('X position - ' + name)
    plt.xlabel('Position [cm]')
    plt.ylabel('#')
    plt.savefig(name + 'X.png')
 
    plt.figure()
    plt.hist(df[["y1"]].transpose(), bins=20, edgecolor='k', alpha=0.7)
    plt.title('Y position - ' + name)
    plt.xlabel('Position [cm]')
    plt.ylabel('#')
    plt.savefig(name + 'Y.png')
  
    plt.figure()
    plt.hist(df[["z1"]].transpose(), bins=20, edgecolor='k', alpha=0.7)
    plt.title('Z position - ' + name)
    plt.xlabel('Position [cm]')
    plt.ylabel('#')
    plt.savefig(name + 'Z.png')

def emissionPoint(row):
    sOfL = 0.03 # cm/us
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

def reconstruction(df):
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')

    for index, row in df.iterrows():
        if (index > 50000): break
        point = emissionPoint(row)
        ax.scatter(xs=point['x'], ys=point['y'], zs=point['z'], c="green", s = 1)

    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    plt.title('NEMA - JPET simulation recostrucion')
    plt.show()

directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/'
fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00'
df, xTrain, xTest, yTrain, yTest = createLearningBatches(directory + fileName)
reconstruction(df[df["class"] == 1])
createStats(df, "NEMA")