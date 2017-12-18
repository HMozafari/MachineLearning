import math

import pandas
import random

from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
# from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

######## Summarize the Dataset ###############
print("Summarize the Dataset")
print(dataset.shape)

# Peek at the Data
print("\n 20 points as the peek for the Data:")
print(dataset.head(20))

# Statistical Summary
print("\n Statistical Summary:")
print(dataset.describe())

# Class Distribution
# Lets now take a look at the number of instances (rows)
# that belong to each class. We can view this as an absolute count.
print(dataset.groupby('class').size())

print("\n \n The size of dataset is: " + str(len(dataset)))


################## Spliting Input set into training ###############
#################  as well as test set. ###########################
def giveIndextoDataFrameSet(dataFrameSet):
    return (pandas.DataFrame(dataFrameSet, index=range(len(dataFrameSet))[1:]))


def splitDataSet(splitPrecentage, dataSet):
    testSet = pandas.DataFrame([{}])
    testSet = testSet.dropna()
    trainingSet = pandas.DataFrame([{}])
    trainingSet = trainingSet.dropna()

    split = splitPrecentage / 100

    for index, row in dataset.iterrows():

        if random.random() < split:
            trainingSet = trainingSet.append(row)
        else:
            testSet = testSet.append(row)

    return (trainingSet, testSet)


def euclidianDistance(dataListInst1, dataListInst2):
    distance = 0
    cnt = 0
    for elem in dataListInst1:
        distance += pow(dataListInst1[cnt] - dataListInst2[cnt], 2)
        cnt = cnt + 1
    return math.sqrt(distance)


def getNeighbors(dataFrameList, clusterCentroid, kFirstNeighbors):
    distances = []
    for index, row in dataFrameList.iterrows():
        dist = euclidianDistance(dataListElem.iloc[row][2:].values.T.tolist(), clusterCentroid)
        df = pd.DataFrame([{'dist': dist}])
        dataFrameList.iloc[row].append(df)
        distances.append(dist)
    distances.sort(key=operator.itemgetter(1))
    return distances


# dataset = list(dataset[1:])
# for x in range(len(dataset)-1):
# for y in range(3):
#   dataset[x][y] = float(dataset[x][y])
# if random.random() < split:
#    trainingSet.append(dataset[x])
# else:
#   testSet.append(dataset[x])
print (giveIndextoDataFrameSet(dataset))

[trainingSet, testSet] = splitDataSet(66.66, dataset)
print("euclidian distance is:")
print(trainingSet.iloc[1])
print(trainingSet.iloc[1][1:].values.T.tolist())
print(euclidianDistance(trainingSet.iloc[1][1:].values.T.tolist(), trainingSet.iloc[2][1:].values.T.tolist(), 0))

print("trainingSet specifications: \n")
print(trainingSet.describe())
trainingSet = trainingSet.dropna()
print("traningSet: \n\n")
print(trainingSet)

print("testSet specifications: \n")
print(testSet.describe())
testSet = testSet.dropna()
print("testSet: \n\n")
print(testSet)

for index, row in testSet.iterrows():
    print("\n")
    print(testSet[index])

for index, row in trainingSet.iterrows():
    print("\n")
    print(trainingSet[index])
