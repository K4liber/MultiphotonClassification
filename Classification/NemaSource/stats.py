from calc import *

directory = '/home/jasiek/Desktop/Studia/PracaMagisterska/Nema_Image_Quality/'
fileName = 'NEMA_IQ_384str_N0_1000_COINCIDENCES_part00'
df, xTrain, xTest, yTrain, yTest = createLearningBatches(directory + fileName)
reconstruction(df[df["class"] == 1])
createStats(df, "NEMA")