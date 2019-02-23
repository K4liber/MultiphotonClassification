import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from calc import *
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn.metrics import confusion_matrix
import pickle

modelName = "XGB10e7"
mkdir_p(modelName)
# Load and transform data into sets 
directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/'
fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00'
df, X_train, X_test, y_train, y_test = createLearningBatches(directory + fileName, 1000)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# fit model on training data
model = XGBClassifier(
    objective = 'binary:logistic', # Logistic regression for binary classification, output probability
    booster = 'gbtree', # Set estimator as gradient boosting tree
    subsample = 1, # Percentage of the training samples used to train (consider this)
)

param_dist = {
    'n_estimators': stats.randint(50, 200), # Number of trees in each classifier
    'learning_rate': stats.uniform(0.15, 0.05), # Contribution of each estimator
    'max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], # Maximum depth of a tree
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1], # The fraction of columns to be subsampled
    'min_child_weight': [1, 2, 3, 4]    # Minimum sum of instance weight (hessian) needed in a child 8
                                        # In linear regression task, this simply corresponds to minimum 
                                        # number of instances needed to be in each node
}

clf = RandomizedSearchCV(
    model,
    param_distributions = param_dist,  
    n_iter = 20, 
    cv = 3, # Cross-validation number of folds
    scoring = 'roc_auc', 
    error_score = 0, 
    verbose = 2, 
    n_jobs = -1
)

clf.fit(X_train, y_train)

# make predictions for test data
y_pred_values = clf.predict(X_test)
y_pred = (y_pred_values > 0.5)

# make predictions for train data
y_pred_values_train = clf.predict(X_train)
y_pred_train = (y_pred_values_train > 0.5)

# Create ROC curves
createROC('XGB-train', y_train, y_pred_values_train, modelName = modelName)
createROC('XGB-test', y_test, y_pred_values, modelName = modelName)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (test): %.2f%%" % (accuracy * 100.0))
accuracyTrain = accuracy_score(y_train, y_pred_train)
print("Accuracy (train): %.2f%%" % (accuracyTrain * 100.0))

# Creating the Confusion Matrixes
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(
    cm, 
    classes=['not pPs', 'pPs'],
    title='XGB-test',
    accuracy='Accuracy: ' + '%.2f' % (accuracy * 100.0) + '%, size: ' + str(y_pred.size),
    modelName = modelName
)
cmTrain = confusion_matrix(y_train, y_pred_train)
plot_confusion_matrix(
    cmTrain, 
    classes=['not pPs', 'pPs'],
    title = 'XGB-train',
    accuracy = 'Accuracy: ' + '%.2f' % (accuracyTrain * 100.0) + '%, size: ' + str(y_pred_train.size),
    modelName = modelName
)

# save best model and all results to file
pickle.dump(clf.best_estimator_, open(modelName + "/bestXGB.dat", "wb"))
pickle.dump(clf.cv_results_, open(modelName + "/CVresults.dat", "wb"))
# plot single tree
fig = plt.figure()
fig.set_size_inches(3600, 2400)
ax = plot_tree(clf.best_estimator_, rankdir='LR')
plt.tight_layout()
plt.savefig(modelName + "/bestTree.png", dpi = 600)

pPsOrginalPositive = X_test[y_test > 0]
pPsOrginalNegative = X_test[y_test == 0]
pPsPredictedPositive = X_test[y_pred]
pPsPredictedNegative = X_test[y_pred == 0]

FP = pd.merge(pPsPredictedPositive,pPsOrginalNegative, how='inner')
TP = pd.merge(pPsPredictedPositive,pPsOrginalPositive, how='inner')
TN = pd.merge(pPsPredictedNegative,pPsOrginalNegative, how='inner')
FN = pd.merge(pPsPredictedNegative,pPsOrginalPositive, how='inner')

saveHistograms(FP, TP, TN, FN, modelName)
reconstructionTest2D(FP, TP, modelName = modelName, title = 'IEC - XGB test recostrucion (TP + FP)')
angleVsTime(FP, TP, TN, FN, modelName)