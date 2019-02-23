import pickle

model = pickle.load(open('XGB10e7/bestXGB.dat', 'rb'))
results = pickle.load(open('XGB10e7/CVresults.dat', 'rb'))

print(model.get_params)
print(results)