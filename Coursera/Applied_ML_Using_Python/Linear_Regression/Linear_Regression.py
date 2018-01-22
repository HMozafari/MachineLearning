import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset', '\t')
#dataset = pd.DataFrame(dataset)

print(dataset.head())
#print(dataset['x'])
print(dataset.iloc[:,0])
#plt.scatter(dataset.iloc[1:,0], dataset.iloc[1:,1])
#plt.show()
X = np.array(dataset.iloc[1:,0])
X = np.column_stack((np.ones((len(X),1)), X.T))
print (X)
y = np.array(dataset.iloc[1:,1])


#print (X - y)
#print (np.transpose(X) * X)
W_star = np.dot(np.dot(1/np.dot(np.transpose(X), X) ,np.transpose(X)), y)
print (W_star)

#Error = 1/2 * (y - np.dot(X, W)).T
#print (Error)

#def model (W, x):
#    np.dot(W,x,y)
#    return y


