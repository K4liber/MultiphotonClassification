import pickle

bestXGB = pickle.load(open('XGB10e7/bestXGB.dat', 'rb'))
results = pickle.load(open('XGB10e7/CVresults.dat', 'rb'))

print(bestXGB.get_params)