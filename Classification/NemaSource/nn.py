from stats.calc import *
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
        Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = 11)
    )
    model.add(
        Dense(output_dim = 8, init = 'uniform', activation = 'relu')
    )
    model.add(
        Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')
    )
    sgd = keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['mean_absolute_error']
    )
    return model

# Load and transform data into sets 
directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/'
fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00'
df, X_train, X_test, y_train, y_test = createLearningBatches(directory + fileName)
classifier = buildNN()

trainCallback = TrainCallback((X_train, y_train))
# Fitting our model 
obj = classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 2, 
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

saveHistograms(X_test, y_test.values, y_pred, "NN")