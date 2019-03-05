#!/usr/bin/env python3.6

import xgboost as xgb
from calc import *
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sys

modelName = "XGB10e7"
# Load and transform data into sets 
def loadData():
    # directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/'
    directory = '/mnt/opt/groups/jpet/NEMA_Image_Quality/3000s/'
    fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00'
    global df, X_train, X_test, y_train, y_test
    df, X_train, X_test, y_train, y_test = createLearningBatches(directory + fileName, 1000)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

# Load and transform data into sets 
loadData()
# Load model
bestXGB = pickle.load(open('XGB10e7/bestXGB.dat', 'rb'))
maxEstimators = 1500
max_depth = sys.argv[1]
bestXGB.set_params(**{'n_estimators': maxEstimators, 'max_depth': max_depth})
# Train and test the model
eval_set  = [( X_train, y_train), ( X_test, y_test)]
results = {}
bestXGB.fit(
    X_train, y_train,
    early_stopping_rounds = 20,
    eval_set = eval_set,
    eval_metric = ["error", "logloss"],
    callbacks = [xgb.callback.record_evaluation(results)]
)

# Make predictions for test data
y_pred = bestXGB.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)

# save model to file
pickle.dump(bestXGB, open(modelName + "/xgbMAX_DEPTH" + str(max_depth) + ".dat", "wb"))

# Plot the results
n = range(maxEstimators)
plt.plot(results['validation_0']['error'], label = "błąd treningowy")
plt.plot(results['validation_1']['error'], label = "błąd testowy")
plt.xlabel("liczba drzew")
plt.ylabel("odsetek błędnie sklasyfikowanych próbek")
plt.title("XGBoost error (max_depth = " + max_depth + ", best accuracy: " + str(accuracy) + ")")
plt.legend(loc = "upper right")
plt.savefig("xgbEstimatorsError" + max_depth + ".png")
plt.clf()

plt.plot(results['validation_0']['logloss'], label = "błąd treningowy")
plt.plot(results['validation_1']['logloss'], label = "błąd testowy")
plt.xlabel("liczba drzew")
plt.ylabel("log loss")
plt.title("XGBoost log loss (max_depth: " + max_depth + ")")
plt.legend(loc = "upper right")
plt.savefig("xgbEstimatorsLoss" + max_depth + ".png")
plt.clf()
