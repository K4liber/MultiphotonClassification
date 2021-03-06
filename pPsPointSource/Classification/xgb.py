import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from calc import loadDataFrames, saveHistograms, plot_confusion_matrix, createROC
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn.metrics import confusion_matrix
import pickle

# Load and transform data into sets 
df, X_train, X_test, y_train, y_test, X_test_with_times = loadDataFrames('data.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# fit model on training data
model = XGBClassifier(objective = 'binary:logistic')
param_dist = {'n_estimators': stats.randint(300, 600),
              'learning_rate': stats.uniform(0.05, 0.1),
              'subsample': stats.uniform(0.6, 0.1),
              'max_depth': [5, 6, 7, 8, 9, 10, 11],
              'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1],
              'min_child_weight': [1, 2, 3, 4]
             }

clf = RandomizedSearchCV(model, 
                         param_distributions = param_dist,  
                         n_iter = 5,
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 2, 
                         n_jobs = -1)

clf.fit(X_train, y_train)

# make predictions for test data
y_pred_values = clf.predict(X_test)
y_pred = (y_pred_values > 0.5)

# make predictions for train data
y_pred_values_train = clf.predict(X_train)
y_pred_train = (y_pred_values_train > 0.5)

# Create ROC curves
createROC('XGB-train', y_train, y_pred_values_train)
createROC('XGB-test', y_test, y_pred_values)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (test): %.2f%%" % (accuracy * 100.0))
accuracyTrain = accuracy_score(y_train, y_pred_train)
print("Accuracy (train): %.2f%%" % (accuracyTrain * 100.0))

# Creating the Confusion Matrixes
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['not pPs', 'pPs'],
    modelName='XGB-test',
    accuracy='Accuracy: ' + '%.2f' % (accuracy * 100.0) + '%, size: ' + str(y_pred.size)
)
cmTrain = confusion_matrix(y_train, y_pred_train)
plot_confusion_matrix(cmTrain, classes=['not pPs', 'pPs'],
    modelName='XGB-train',
    accuracy='Accuracy: ' + '%.2f' % (accuracyTrain * 100.0) + '%, size: ' + str(y_pred_train.size)
)

# save model to file
pickle.dump(clf.best_estimator_, open("stats/bestXGB.dat", "wb"))
# plot single tree
plot_tree(clf.best_estimator_, rankdir='LR')
plt.show()

# saveHistograms(X_test_with_times, y_test, y_pred, "XGB")
