import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math
from sklearn.metrics import roc_curve, auc
import numpy as np
import itertools

dataFrameNames = [
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

def createLearningBatches(fileName, size):
    df = pd.read_csv(fileName, sep = "\t", names = dataFrameNames).head(size)
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

def reconstructionTest3D(FP, TP):
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')

    for index, row in TP.iterrows():
        point = emissionPoint(row)
        ax.scatter(
            xs=point['x'], ys=point['y'], zs=point['z'], c="green", s = 1,
            label='TP' if index == TP.first_valid_index() else ""
        )

    for index, row in FP.iterrows():
        point = emissionPoint(row)
        ax.scatter(
            xs=point['x'], ys=point['y'], zs=point['z'], c="red", s = 1,
            label='FP' if index == FP.first_valid_index() else ""
        )

    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    ax.legend(loc='lower left')
    plt.title('JPET NEMA - NN test recostrucion')
    plt.show()

def reconstructionTest2D(FP, TP, title):
    points = pd.DataFrame(columns=['X', 'Y'])

    for index, row in TP.iterrows():
        point = emissionPoint(row)
        points = points.append({'X': point['x'], 'Y': point['y']}, ignore_index = True)

    for index, row in FP.iterrows():
        point = emissionPoint(row)
        points = points.append({'X': point['x'], 'Y': point['y']}, ignore_index = True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.hist2d(points["X"], points["Y"], bins=(200, 200), cmap = plt.cm.jet)
    plt.colorbar()
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    plt.title(title)
    plt.show()

def plot_confusion_matrix(cm, classes, modelName, accuracy, cmap=plt.cm.Blues):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(
        modelName + '\n' +
        accuracy + '\n' + 
        "Precision: " + '%.2f' % (cm[1, 1]*100/(cm[1, 1] + cm[0, 1])) + '% ,'
        "recall: " + '%.2f' % (cm[1, 1]*100/(cm[1, 1] + cm[1, 0])) + '%'
    )
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(modelName + 'confMatrix.png')

def createROC(modelName, y, y_pred):
    fpr_keras, tpr_keras, _ = roc_curve(y, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.clf()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(modelName + '-ROC')
    plt.legend(loc='best')
    plt.savefig(modelName + '-ROC.png')

def saveHistograms(FP, TP, TN, FN, modelName):
    FPStatsFrame = FP[["e1","x1", "y1", "z1", "dt"]].drop_duplicates()
    createStats(FPStatsFrame, modelName + '-False Positive')

    TPStatsFrame = TP[["e1","x1", "y1", "z1", "dt"]].drop_duplicates()
    createStats(TPStatsFrame, modelName + '-True Positive')

    TNStatsFrame = TN[["e1","x1", "y1", "z1", "dt"]].drop_duplicates()
    createStats(TNStatsFrame, modelName + '-True Negative')

    FNStatsFrame = FN[["e1","x1", "y1", "z1", "dt"]].drop_duplicates()
    createStats(FNStatsFrame, modelName + '-False Negative')