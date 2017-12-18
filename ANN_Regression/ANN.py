import numpy
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
#import sklearn.model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import optimize
from scipy.optimize import basinhopping

################################
#THIS CODE IS INSPIRED FROM THIS WEB-SITE:
#### http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/ ####
###############################

class MyBounds(object):
     def __init__(self, xmax=[20,20], xmin=[2,2] ):
         self.xmax = np.array(xmax)
         self.xmin = np.array(xmin)
     def __call__(self, **kwargs):
         x = kwargs["x_new"]
         tmax = bool(np.all(x <= self.xmax))
         tmin = bool(np.all(x >= self.xmin))
         return tmax and tmin

class MyTakeStep(object):
	def __init__(self, stepsize=1):
		self.stepsize = stepsize
	def __call__(self, x):
		s = self.stepsize
		for i in range(0, len(x)):
			randNum=int(np.random.uniform(0, 3))
			if ( randNum<3 and randNum >=2):
				x[i] = int(x[i]) + s
			elif (randNum <2 and randNum >=1):
				x[i] = int(x[i]) - 0
			elif (randNum < 1 and randNum >= 0):
				x[i] = int(x[i]) - s
			print(x, randNum)


		# x[0] += s * int(np.random.uniform(0, 1))
		# x[1:] += np.random.uniform(-s, s, x[1:].shape)
		return x

def main ():
	minimizer_kwargs = {"method": "L-BFGS-B", "jac": True}
	x0 = [3, 4]

	# mlp_model_run(3, 4)
	#ret = basinhopping(mlp_model_run, x0, stepsize=1, minimizer_kwargs=minimizer_kwargs, niter = 200)
	mytakestep = MyTakeStep()
	mybounds = MyBounds()
	ret = basinhopping(func2d, x0, take_step=mytakestep, accept_test=mybounds, stepsize=1, minimizer_kwargs=minimizer_kwargs, niter=200, interval=50)
	print("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],	ret.x[1], ret.fun))



def func2d(x):
     f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
     df = np.zeros(2)
     df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
     df[1] = 2. * x[1] + 0.2
     return f, df


def mlp_model_run(x):
	nupl1 = x[0]
	nupl2 = x[1]
	dataframe = pandas.read_csv("housing.data", delim_whitespace=True, header=None)  # read dataSet
	dataset = dataframe.values  # changes the datafram to array of numbers
	# split into input (X) and output (Y) variables
	X = dataset[:, 0:13]  # there are 13 inputs X
	Y = dataset[:, 13]  # there are one output y
	seed = 7

	numpy.random.seed(seed)
	# evaluate model with standardized dataset
	estimator = KerasRegressor(build_fn=baseline_model, nupl1=nupl1, nupl2=nupl2, nb_epoch=100, batch_size=5, verbose=0)

	kfold = KFold(n_splits=10, random_state=seed)  # it breaks input set into 10 parts and
	# it dedicate 1/10 to\ validation set and
	# 9/10 to training set.
	# however, it select these training as well as validation set many times and then trains estimator.
	# cross_val_score(estimator, X, Y, cv=kfold)
	results = cross_val_score(estimator, X, Y, cv=kfold)  # it performs cross validation by
	# checking the estimator output
	# against validation set
	# This result contains all stochastic parameters for this comaprison such as mean, standard deviation,
	print("Results: %.2f MSE with 95 precent confidence interval (%.2f)" % (results.mean(), results.std()))
	return results.mean()

	# run_mlp_model(X, Y)


	# params = (X, Y)
	# np.random.seed(seed)
	# x0 = np.array([10, 13, 0, 0, 0])
	# res = optimize.basinhopping(run_mlp_model, x0, args=params, schedule='boltzmann',
	# 					  full_output=True, maxiter=500, lower=-10,
	# 					  upper=10, dwell=250, disp=True)
	# print res

	#run_mlp_model(z, *params)

def baseline_model(nupl1,nupl2):
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, init='normal', activation='relu')) # define a hidden layer with 13 nodes, while the input nodes are 13.
									     # use  rectifieractivation function for the second layer.
	model.add(Dense(nupl1, init='normal'))
	model.add(Dense(nupl2, init='normal'))
	model.add(Dense(1, init='normal')) # define another layer with one input (output layer) that has just one node and its activation function is sum (default).
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model




#
# ###########################################################################
# ############# STANDARD-RAZATION OF INPUT SET BEFORE TRAINING ##############
# ###########################################################################
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized Results: %.2f MSE with 95 percent confidence interval (%.2f)" % (results.mean(), results.std()))

###########################################################################
##################### ADDING MORE LAYERS TO ANN ###########################
###########################################################################

# def larger_model():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
# 	model.add(Dense(10, init='normal', activation='relu'))
# 	model.add(Dense(1, init='normal'))
# 	# Compile model
# 	model.compile(loss='mean_squared_error', optimizer='adam')
# 	return model
#
# def run_mlp_model(X, Y):
#
# 	seed =7
# 	np.random.seed(seed)
# 	estimators = []
# 	estimators.append(('standardize', StandardScaler()))
# 	estimators.append(('mlp', KerasRegressor(build_fn=larger_model, x=[2, 3], nb_epochs=100, batch_size=5, verbose=0)))
# 	pipeline = Pipeline(estimators)
# 	kfold = KFold(n_splits=10, random_state=seed)
# 	results = cross_val_score(pipeline, X, Y, cv=kfold)
# 	print("Larger Net. Results: %.2f MSE with 95 percent confidence interval (%.2f)" % (results.mean(), results.std()))
# 	return results.mean()

if __name__ == '__main__':
	main()
