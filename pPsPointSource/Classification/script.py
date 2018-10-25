import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from calc import emissionPoint, loadDataFrames, reconstruction, createHistograms, saveHistograms

# Load and transform data into sets 
df, X_train, X_test, y_train, y_test, X_test_with_times = loadDataFrames('data.csv')

# Initializing Neural Network
classifier = Sequential()

# Adding the first hidden layer
classifier.add(
    Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = 11)
)

# Adding the second hidden layer
classifier.add(
    Dense(output_dim = 6, init = 'uniform', activation = 'relu', )
)

# Adding the output layer
classifier.add(
    Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')
)

# Compiling Neural Network
classifier.compile(
    optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']
)

# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 2)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Creating the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# reconstruction(FP, TP, TN, FN)

# Stats for all particles considered
allStatsFrame = df[["EventID1","TrackID1","e1","x1", "y1", "z1", "dt"]] \
                    .drop_duplicates()
createHistograms(allStatsFrame, 'all')

# Stats for pPs events
pPsStatsFrame = df[["EventID1","TrackID1","e1","x1", "y1", "z1", "dt"]] \
                    .loc[df['pPs'] == 1] \
                    .drop_duplicates()
createHistograms(pPsStatsFrame, 'pPs')

# Stats for not pPs events
notpPsStatsFrame = df[["EventID1","TrackID1","e1","x1", "y1", "z1", "dt"]] \
                    .loc[df['pPs'] == 0] \
                    .drop_duplicates()
createHistograms(notpPsStatsFrame, 'notpPs')

print(y_test.values)

saveHistograms(X_test_with_times, y_test.values, y_pred, "NN")
