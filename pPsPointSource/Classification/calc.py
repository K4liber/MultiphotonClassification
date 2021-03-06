import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
from sklearn.metrics import roc_curve, auc

sOfL = 300 # mm/ns
names = [
    "evID1", "evID2", "trID1", "trID2", "x1", "y1", "z1", "x2", "y2", "z2",
    "e1", "e2", "dt", "t1", "t2", "vol1", "vol2", "pPs"
]

def emissionPoint(row):
    x1 = row['x1']
    y1 = row['y1']
    z1 = row['z1']
    x2 = row['x2']
    y2 = row['y2']
    z2 = row['z2']
    dt = row['dt']
    halfX = (x1 - x2)/2
    halfY = (y1 - y2)/2
    halfZ = (z1 - z2)/2
    LORHalfSize = math.sqrt(halfX**2 + halfY**2 + halfZ**2)
    versX = halfX/LORHalfSize
    versY = halfY/LORHalfSize
    versZ = halfZ/LORHalfSize
    dX = dt*sOfL*versX/2
    dY = dt*sOfL*versY/2
    dZ = dt*sOfL*versZ/2
    return { 
        'x':(x1+x2)/2 - dX,
        'y':(y1+y2)/2 - dY,
        'z':(z1+z2)/2 - dZ,
    }

def centerDistance(row):
    rec = emissionPoint(row)
    return (math.sqrt(rec['x']**2+rec['y']**2+rec['z']**2))

def energySum(row):
    return (row['e1'] + row['e2'])

def loadDataFrames(filename):
    codes = {'detector1':1, 'detector2':2, 'detector3':3}
    # codes = {'box_with_water':1, 'crystal1':2, 'crystal2':3, 'crystal3':4}
    df = pd.read_csv(filename, names = names)
    df['dt'] = df['t1'] - df['t2']
    df['dist'] = df.apply(lambda row: centerDistance(row),axis=1)
    df['energySum'] = df.apply(lambda row: energySum(row),axis=1)
    df['vol1'] = df['vol1'].map(codes)
    df['vol2'] = df['vol2'].map(codes)
    X = df.drop(["pPs", "evID1", "evID2", "trID1", "trID2", "t1", "t2"], axis=1)
    y = df[["pPs"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_test_with_times = X_test.copy()
    return df, X_train, X_test, y_train, y_test, X_test_with_times

def reconstruction(FP, TP, TN, FN):
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')

    for index, row in TP.iterrows():
        point = emissionPoint(row)
        ax.scatter(
            xs=point['x'], ys=point['y'], zs=point['z'], c="green",
            label='TP' if index == TP.first_valid_index() else ""
        )

    for index, row in FP.iterrows():
        point = emissionPoint(row)
        ax.scatter(
            xs=point['x'], ys=point['y'], zs=point['z'], c="red",
            label='FP' if index == FP.first_valid_index() else ""
        )


    for index, row in FN.iterrows():
        point = emissionPoint(row)
        ax.scatter(
            xs=point['x'], ys=point['y'], zs=point['z'], c="blue",
            label='FN' if index == FN.first_valid_index() else ""
        )

    for index, row in TN.iterrows():
        point = emissionPoint(row)
        ax.scatter(
            xs=point['x'], ys=point['y'], zs=point['z'], c="yellow",
            label='TN' if index == TN.first_valid_index() else ""
        )

    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')
    ax.legend(loc='lower left')
    plt.title('pPs point source - JPET simulation recostrucion')
    plt.show()

def createHistograms(df, name):
    plt.figure()
    plt.hist(df[["e1"]].transpose(), bins=40, edgecolor='k', alpha=0.7)
    e1Mean = df["e1"].mean()
    plt.axvline(e1Mean, color='k', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    plt.text(   e1Mean + e1Mean/10, 
                max_ - max_/10, 
                'Mean: {:.2f}'.format(e1Mean))
    plt.title('Energy loss - ' + name)
    plt.xlabel('Energy [keV]')
    plt.ylabel('#')
    plt.savefig('stats/' + name + 'Energy.png')

    plt.figure()
    plt.hist(df[["dt"]].transpose(), bins=20, edgecolor='k', alpha=0.7)
    dtMean = df["dt"].mean()
    plt.axvline(dtMean, color='k', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    plt.text(   dtMean + abs(dtMean/10), 
                max_ - max_/10, 
                'Mean: {:.2f}'.format(dtMean))
    plt.title('Detection time difference - ' + name)
    plt.xlabel('time difference [ns]')
    plt.ylabel('#')
    plt.savefig('stats/' + name + 'Time.png')

    plt.figure()
    plt.hist(df[["x1"]].transpose(), bins=20, edgecolor='k', alpha=0.7)
    x1Mean = df["x1"].mean()
    plt.axvline(x1Mean, color='k', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    plt.text(   x1Mean + abs(x1Mean/10), 
                max_ - max_/10, 
                'Mean: {:.2f}'.format(x1Mean))
    plt.title('X position - ' + name)
    plt.xlabel('Position [mm]')
    plt.ylabel('#')
    plt.savefig('stats/' + name + 'X.png')
 
    plt.figure()
    plt.hist(df[["y1"]].transpose(), bins=20, edgecolor='k', alpha=0.7)
    y1Mean = df["y1"].mean()
    plt.axvline(y1Mean, color='k', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    plt.text(   y1Mean + abs(y1Mean/10), 
                max_ - max_/10, 
                'Mean: {:.2f}'.format(y1Mean))
    plt.title('Y position - ' + name)
    plt.xlabel('Position [mm]')
    plt.ylabel('#')
    plt.savefig('stats/' + name + 'Y.png')
  
    plt.figure()
    plt.hist(df[["z1"]].transpose(), bins=20, edgecolor='k', alpha=0.7)
    z1Mean = df["z1"].mean()
    plt.axvline(z1Mean, color='k', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    plt.text(   z1Mean + abs(z1Mean/10), 
                max_ - max_/10, 
                'Mean: {:.2f}'.format(z1Mean))
    plt.title('Z position - ' + name)
    plt.xlabel('Position [mm]')
    plt.ylabel('#')
    plt.savefig('stats/' + name + 'Z.png')

def saveHistograms(X_test_with_times, y_test, y_pred, modelName):
    pPsOrginalPositive = X_test_with_times[y_test > 0]
    pPsOrginalNegative = X_test_with_times[y_test == 0]
    pPsPredictedPositive = X_test_with_times[y_pred]
    pPsPredictedNegative = X_test_with_times[y_pred == 0]

    FP = pd.merge(pPsPredictedPositive,pPsOrginalNegative, how='inner')
    TP = pd.merge(pPsPredictedPositive,pPsOrginalPositive, how='inner')
    TN = pd.merge(pPsPredictedNegative,pPsOrginalNegative, how='inner')
    FN = pd.merge(pPsPredictedNegative,pPsOrginalPositive, how='inner')

    reconstruction(FP, TP, TN, FN)

    FPStatsFrame = FP[["e1","x1", "y1", "z1", "dt"]].drop_duplicates()
    createHistograms(FPStatsFrame, modelName + '-False Positive')

    TPStatsFrame = TP[["e1","x1", "y1", "z1", "dt"]].drop_duplicates()
    createHistograms(TPStatsFrame, modelName + '-True Positive')

    TNStatsFrame = TN[["e1","x1", "y1", "z1", "dt"]].drop_duplicates()
    createHistograms(TNStatsFrame, modelName + '-True Negative')

    FNStatsFrame = FN[["e1","x1", "y1", "z1", "dt"]].drop_duplicates()
    createHistograms(FNStatsFrame, modelName + '-False Negative')

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
    plt.savefig('stats/' + modelName + 'confMatrix.png')

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
    plt.savefig('stats/' + modelName + '-ROC.png')