#!/usr/bin/env python3.6

import xgboost as xgb
from calc import *
import pickle
import matplotlib.pyplot as plt

# Load and transform data into sets 
def loadData():
    # directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/'
    directory = '/mnt/opt/groups/jpet/NEMA_Image_Quality/3000s/'
    fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00'
    global df, X_train, X_test, y_train, y_test
    df, X_train, X_test, y_train, y_test = createLearningBatches(directory + fileName, 10000000)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

# Load and transform data into sets 
loadData()
# Load model
bestXGB = pickle.load(open('XGB10e7/bestXGB.dat', 'rb'))
maxEstimators = 300
bestXGB.set_params(**{'n_estimators':maxEstimators})
# Train and test the model
eval_set  = [( X_train, y_train), ( X_test, y_test)]
results = {}
bestXGB.fit(
    X_train, y_train, 
    eval_set = eval_set,
    callbacks = [xgb.callback.record_evaluation(results)]
)
# Plot the results
n = range(maxEstimators)
plt.plot(n, results['validation_0']['error'], label = "błąd treningowy")
plt.plot(n, results['validation_1']['error'], label = "błąd testowy")
plt.xlabel("liczba drzew")
plt.ylabel("odsetek błędnie sklasyfikowanych próbek")
plt.title("XGBoost - bład predykcji w funkcji liczby estymatorów")
plt.legend(loc = "upper right")
plt.savefig("xgbEstimators.png")
