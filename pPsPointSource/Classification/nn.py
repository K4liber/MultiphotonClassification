import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from calc import createHistograms, saveHistograms, plot_confusion_matrix
from calc import loadDataFrames, reconstruction, createROC

class TrainCallback(Callback):
    def __init__(self, train_data):
        self.train_data = train_data
        self.acc = 0
        self.cm = None
        self.y_pred_values = None

    def on_train_end(self, epoch, logs={}):
        x_train, y_train = self.train_data
        self.y_pred_values = self.model.predict(x_train)
        y_pred = (self.y_pred_values > 0.5)
        self.acc = accuracy_score(y_train, y_pred)
        self.cm = confusion_matrix(y_train, y_pred)
        print('\nTraining acc: {}\n'.format(self.acc))

def buildNN():
    model = Sequential()
    model.add(
        Dense(output_dim = 14, init = 'uniform', activation = 'relu', input_dim = 15)
    )
    model.add(
        Dense(output_dim = 10, init = 'uniform', activation = 'relu')
    )
    model.add(
        Dense(output_dim = 8, init = 'uniform', activation = 'relu')
    )
    model.add(
        Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')
    )
    sgd = keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']
    )
    return model

# Load and transform data into sets 
df, X_train, X_test, y_train, y_test, X_test_with_times = loadDataFrames('data.csv')
print(df.head())
classifier = buildNN()

trainCallback = TrainCallback((X_train, y_train))
# Fitting our model 
obj = classifier.fit(X_train, y_train, batch_size = 16, nb_epoch = 10, 
    callbacks=[trainCallback])

# Predicting the Test set results
y_pred_values = classifier.predict(X_test)
y_pred = (y_pred_values > 0.5)

# Create ROC curves
createROC('NN-test', y_test, y_pred_values)
createROC('NN-train', y_train, trainCallback.y_pred_values)

# Creating the Confusion Matrix
cmTest = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Accuracy (train): %.2f%%" % (trainCallback.acc * 100.0))

'''
plot_confusion_matrix(cmTest, classes=['not pPs', 'pPs'],
    modelName='NN-test',
    accuracy='Accuracy: ' + '%.2f' % (accuracy * 100.0) + 
    "%, size: " + str(y_pred.size)
)
plot_confusion_matrix(trainCallback.cm, classes=['not pPs', 'pPs'],
    modelName='NN-train',
    accuracy='Accuracy: ' + '%.2f' % (trainCallback.acc * 100.0) + 
    "%, size: " + str(y_train.size)
)
'''

# Stats for all particles considered
# allStatsFrame = df[["EventID1","TrackID1","e1","x1", "y1", "z1", "dt"]].drop_duplicates()
# createHistograms(allStatsFrame, 'all')

# Stats for pPs events
# pPsStatsFrame = df[["EventID1","TrackID1","e1","x1", "y1", "z1", "dt"]].loc[df['pPs'] == 1].drop_duplicates()
# createHistograms(pPsStatsFrame, 'pPs')

# Stats for not pPs events
# notpPsStatsFrame = df[["EventID1","TrackID1","e1","x1", "y1", "z1", "dt"]].loc[df['pPs'] == 0].drop_duplicates()
# createHistograms(notpPsStatsFrame, 'notpPs')

saveHistograms(X_test_with_times, y_test.values, y_pred, "NN")
