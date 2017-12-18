import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 7 # let us have the same random numbers when for each run of this program
numpy.random.seed(seed)

### Read data from DataSer ####
dataframe = pandas.read_csv("../data/sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)	# 60 float number as the strength of a sonar result on a piece of metal or stone cylander
Y = dataset[:,60] # tell you if it is metal or stone
###############################
	
# encode class values as integers
# change the output string values to integer numbers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
###############################

### To use Keras models with scikit-learn, we must use the KerasClassifier wrapper. 
### This class takes a function that creates and returns our neural network model.
### It also takes arguments that it will pass along to the call to fit() such as the number of epochs and the batch size.

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, init='normal', activation='relu')) ### input layer(60 nodes)  + 1st hidden layer(60 nodes)
	model.add(Dense(1, init='normal', activation='sigmoid')) ### output layer
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # define cost function as well as the optimization function (e.g., gradient decent).
	return model

#evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



############# PHASE 2 STANDARD-IZAITION OF THE INPUTS ##############
# An effective data preparation scheme for tabular data when building neural network models is standardization.
# This is where the data is rescaled such that the mean value for each attribute is 0 and the standard deviation is 1.
# This preserves Gaussian and Gaussian-like distributions whilst normalizing the central tendencies for each attribute.
# Rather than performing the standardization on the entire dataset, it is good practice to train the standardization 
# procedure on the training data within the pass of a cross validation run and to use the trained standardization to 
# prepare the “unseen” test fold. This makes standardization a step in model preparation in the cross validation process 
# and it prevents the algorithm having knowledge of “unseen” data during evaluation, knowledge that might be passed from the data preparation scheme like a crisper distribution.

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


##### Reducing the network size in terms of hidden units per layer #####
##### when there are not many features in an input data set, we reduce the size of hidden-layers ###
##### in this way we can prevent out model from 
def create_smaller():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=60, init='normal', activation='relu')) # input layer (60 nodes) + 1st hidden layer(30 nodes)
	model.add(Dense(1, init='normal', activation='sigmoid')) #output layer 1 node
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


#### larger model
#### larger model (a more number of hidden layers), let the NN to combine different features that are extracted in the 
#### first layer and extract more complicated (combined) features 
def create_larger():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=30, init='normal', activation='relu'))
	model.add(Dense(30, init='normal', activation='relu'))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

