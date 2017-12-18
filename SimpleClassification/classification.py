#!/usr/bin/env python3
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
#from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

######## Summarize the Dataset ###############
print ("Summarize the Dataset")
print(dataset.shape)

# Peek at the Data
print ("\n 20 points as the peek for the Data:")
print(dataset.head(20))

# Statistical Summary
print ("\n Statistical Summary:")
print(dataset.describe())

# Class Distribution
# Lets now take a look at the number of instances (rows)
# that belong to each class. We can view this as an absolute count.
print(dataset.groupby('class').size())

#################################################
#################################################

############ Data Visualization #################
# This gives us a much clearer idea 
# of the distribution of the input attributes:
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# We can also create a histogram of each input
# variable to get an idea of the distribution.
dataset.hist()
plt.show()

# Multivariate Plots
# Now we can look at the interactions between the variables.
scatter_matrix(dataset)
plt.show()

#################################################
#################################################

##########Evaluate Some Algorithms ##############
# Here is what we are going to cover in this step:

 #    Separate out a validation dataset.
 #    Set-up the test harness to use 10-fold cross validation.
 #    Build 5 different models to predict species from flower measurements
 #    Select the best model.

### Create a Validation Dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
#validation_size = 0.20

#seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#################################################
#################################################

############### Build Models ####################

# We don't know which algorithms would be good on this problem or what configurations to use. We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good results.

# Let's evaluate 6 different algorithms:

#     Logistic Regression (LR)
#     Linear Discriminant Analysis (LDA)
#     K-Nearest Neighbors (KNN).
#     Classification and Regression Trees (CART).
#     Gaussian Naive Bayes (NB).
#     Support Vector Machines (SVM).

# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable.

# Let's build and evaluate our five models:




models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

	############# Select Best Model ###############

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

######################################################
######################################################


###############  Make Predictions ####################

# The KNN algorithm was the most accurate model that we tested. 
# Now we want to get an idea of the accuracy of the model on our validation set.

# This will give us an independent final check on the accuracy of the best model. 
# It is valuable to keep a validation set just in case you made a slip during training, 
# such as overfitting to the training set or a data leak.
# Both will result in an overly optimistic result.

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
