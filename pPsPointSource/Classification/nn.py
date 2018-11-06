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
from calc import emissionPoint, loadDataFrames, reconstruction, createHistograms, saveHistograms, plot_confusion_matrix

class TrainCallback(Callback):
    def __init__(self, train_data):
        self.train_data = train_data
        self.acc = 0
        self.cm = None

    def on_train_end(self, epoch, logs={}):
        x_train, y_train = self.train_data
        y_pred = self.model.predict(x_train)
        y_pred = (y_pred > 0.5)
        self.acc = accuracy_score(y_train, y_pred)
        self.cm = confusion_matrix(y_train, y_pred)
        print('\nTraining acc: {}\n'.format(self.acc))


# Load and transform data into sets 
df, X_train, X_test, y_train, y_test, X_test_with_times = loadDataFrames('data.csv')

# Initializing Neural Network
classifier = Sequential()

# Adding the first hidden layer
classifier.add(
    Dense(output_dim = 14, init = 'uniform', activation = 'relu', input_dim = 12)
)

# Adding the first hidden layer
classifier.add(
    Dense(output_dim = 10, init = 'uniform', activation = 'relu')
)

# Adding the second hidden layer
classifier.add(
    Dense(output_dim = 8, init = 'uniform', activation = 'relu')
)

# Adding the output layer
classifier.add(
    Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')
)

sgd = keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
# Compiling Neural Network
classifier.compile(
    optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']
)

trainCallback = TrainCallback((X_train, y_train))
# Fitting our model 
obj = classifier.fit(X_train, y_train, batch_size = 16, nb_epoch = 1000, 
    callbacks=[trainCallback])

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Creating the Confusion Matrix
cmTest = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Accuracy (train): %.2f%%" % (trainCallback.acc * 100.0))
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

# Stats for all particles considered
allStatsFrame = df[["EventID1","TrackID1","e1","x1", "y1", "z1", "dt"]] \
                    .drop_duplicates()
# createHistograms(allStatsFrame, 'all')

# Stats for pPs events
pPsStatsFrame = df[["EventID1","TrackID1","e1","x1", "y1", "z1", "dt"]] \
                    .loc[df['pPs'] == 1] \
                    .drop_duplicates()
# createHistograms(pPsStatsFrame, 'pPs')

# Stats for not pPs events
notpPsStatsFrame = df[["EventID1","TrackID1","e1","x1", "y1", "z1", "dt"]] \
                    .loc[df['pPs'] == 0] \
                    .drop_duplicates()
# createHistograms(notpPsStatsFrame, 'notpPs')

# saveHistograms(X_test_with_times, y_test.values, y_pred, "NN")
