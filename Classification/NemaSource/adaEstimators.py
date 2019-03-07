#!/usr/bin/env python3.6

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from calc import *
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sys

def loadData():
    # directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/'
    directory = '/mnt/opt/groups/jpet/NEMA_Image_Quality/3000s/'
    fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00'
    global df, X_train, X_test, y_train, y_test
    df, X_train, X_test, y_train, y_test = createLearningBatches(directory + fileName, 10000000)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

modelName = "ADA10e7"
loadData()
max_depth = int(sys.argv[1])
n_estimators = 1500
model = AdaBoostClassifier(
    base_estimator = DecisionTreeClassifier(max_depth = max_depth),
    n_estimators = n_estimators,
    learning_rate = 0.2
)
model.fit(X_train, y_train)

test_errors = []
train_errors = []

for test_predicts, train_predicts in zip(
        model.staged_predict(X_test), model.staged_predict(X_train)):
    test_errors.append(
        1. - accuracy_score(test_predicts, y_test))
    train_errors.append(
        1. - accuracy_score(train_predicts, y_train))

# save model to file
pickle.dump(model, open(modelName + "/adaEstimators" + str(n_estimators) + "Depth" + str(max_depth) + ".dat", "wb"))

bestAccuracy = 1.0 - min(test_errors)
bestNEstimators = test_errors.index(min(test_errors))
# Plot the results
plt.plot(train_errors, label = "błąd treningowy")
plt.plot(test_errors, label = "błąd testowy")
plt.xlabel("liczba drzew")
plt.ylabel("odsetek błędnie sklasyfikowanych próbek")
plt.title("AdaBoost error (max_depth = " + str(max_depth) + ", best test accuracy: " + str(bestAccuracy) + ", n = " + str(bestNEstimators) + ")")
plt.legend(loc = "upper right")
plt.savefig("adaEstimators" + str(n_estimators) + "Depth" + str(max_depth) + ".png")
plt.clf()
