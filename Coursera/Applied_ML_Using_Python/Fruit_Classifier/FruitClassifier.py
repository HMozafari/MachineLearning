import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_table('Fruit_data_with_colors.txt')

print(fruits.head())

# holds the features of the fruits dataset
X = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

knn = KNeighborsClassifier(n_neighbors= 5)
knn.fit(X_train, y_train)

print("The accuracy of KNN over test set is" + str(knn.score(X_test, y_test)))

print ("knn prediction about an example is", knn.predict([5.5, 2.2,10, 0.70]) )
