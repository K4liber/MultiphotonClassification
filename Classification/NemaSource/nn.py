from calc import *
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, confusion_matrix

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
        Dense(
            output_dim = 15, # Layer size
            init = 'uniform', # Way to set the initial random weights
            activation = 'relu', # Activation function
            input_dim = 11 # Number of attributes (input layer size)
        ) 
    )
    model.add(
        Dense(output_dim = 8, init = 'uniform', activation = 'relu')
    )
    model.add(
        Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')
    )
    # sgd = keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        optimizer = 'adam', # lr=0.001,
                            # beta_1=0.9, 
                            # beta_2=0.999, 
                            # epsilon=None, 
                            # decay=0.0, 
                            # amsgrad=False
        loss = 'binary_crossentropy', # The function that will get minimized by the optimizer.
        metrics = ['mean_absolute_error'] # The metric used to judge the performance of your model.
    )
    return model

dataSize = int(sys.argv[1])
modelName = "NN" + str(dataSize)
mkdir_p(modelName)

# Load and transform data into sets 
directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/'
fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00'
df, X_train, X_test, y_train, y_test = createLearningBatches(directory + fileName, 10000000)

classifier = buildNN()
trainCallback = TrainCallback((X_train, y_train))

# Fitting our model 
obj = classifier.fit(
    X_train, y_train, 
    batch_size = 32, # Number of sample use to training at the the time
    nb_epoch = 3, # Repeating the use of training data
    callbacks=[trainCallback]
)

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

# Save histograms with stats
pPsOrginalPositive = X_test[y_test.values > 0]
pPsOrginalNegative = X_test[y_test.values == 0]
pPsPredictedPositive = X_test[y_pred]
pPsPredictedNegative = X_test[y_pred == 0]

FP = pd.merge(pPsPredictedPositive,pPsOrginalNegative, how='inner')
TP = pd.merge(pPsPredictedPositive,pPsOrginalPositive, how='inner')
TN = pd.merge(pPsPredictedNegative,pPsOrginalNegative, how='inner')
FN = pd.merge(pPsPredictedNegative,pPsOrginalPositive, how='inner')

saveHistograms(FP, TP, TN, FN, "NN")
reconstructionTest2D(FP, TP, modelName = "NN", title = 'IEC - NN test recostrucion (TP + FP)')