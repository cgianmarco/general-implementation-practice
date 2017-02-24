import numpy as np 
import pandas as pd 


# Preprocessing

ds = pd.read_csv('car.data', sep=",", header=None)
ds.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Class']

ds['buying'].replace('vhigh', 1, inplace=True)
ds['buying'].replace('high', 2, inplace=True)
ds['buying'].replace('med', 3, inplace=True)
ds['buying'].replace('low', 4, inplace=True)


ds['maint'].replace('vhigh', 1, inplace=True)
ds['maint'].replace('high', 2, inplace=True)
ds['maint'].replace('med', 3, inplace=True)
ds['maint'].replace('low', 4, inplace=True)

ds['doors'].replace('2', 1, inplace=True)
ds['doors'].replace('3', 2, inplace=True)
ds['doors'].replace('4', 3, inplace=True)
ds['doors'].replace('5more', 4, inplace=True)

ds['persons'].replace('2', 1, inplace=True)
ds['persons'].replace('4', 2, inplace=True)
ds['persons'].replace('more', 3, inplace=True)

ds['lug_boot'].replace('small', 1, inplace=True)
ds['lug_boot'].replace('med', 2, inplace=True)
ds['lug_boot'].replace('big', 3, inplace=True)

ds['safety'].replace('low', 1, inplace=True)
ds['safety'].replace('med', 2, inplace=True)
ds['safety'].replace('high', 3, inplace=True)

ds['Class'].replace('unacc', 1, inplace=True)
ds['Class'].replace('acc', 2, inplace=True)
ds['Class'].replace('good', 3, inplace=True)
ds['Class'].replace('vgood', 4, inplace=True)

ds = ds[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Class']]



# Normalization

def normalize(column):
	return (column - column.mean()) / (column.max() - column.min())

buying = normalize(ds['buying'])
maint = normalize(ds['maint'])
doors = normalize(ds['doors'])
persons = normalize(ds['persons'])
lug_boot = normalize(ds['lug_boot'])
safety = normalize(ds['safety'])
Class = normalize(ds['Class'])



X = np.asarray(pd.concat([buying, maint, doors, persons, lug_boot, safety], axis=1, join='inner'))
Y = np.asarray(pd.get_dummies(ds['Class']))

print X[:10]
print Y[:10]

print X.shape
print Y.shape


################################################################################

# We use a 2 layers neural network with one 8 neurons hidden layer

n_inputs = 6
n_h1 = 8
n_outputs = 4
n_samples = len(X)



w_l1 = np.random.randn(n_inputs, n_h1)
w_output = np.random.randn(n_h1, n_outputs)

h1 = np.tanh(X.dot(w_l1))
assert h1.shape == (n_samples, n_h1), "Error on first layer"

prediction = h1.dot(w_output)
assert prediction.shape == (n_samples, n_outputs), "Error on second layer"

# TODO: add softmax












