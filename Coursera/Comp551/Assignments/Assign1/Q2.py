import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt

from Q1P1 import W_star, model, mean_squared_error, mean_squared_error_with_regu, add_poly_features



def main():
    reg_model_poly_degree = 1 #linear model

    Dataset_2_train = pd.read_csv("hwk1_datasets/Datasets/Dataset_2_train.csv")
    Dataset_2_valid = pd.read_csv("hwk1_datasets/Datasets/Dataset_2_valid.csv")
    Dataset_2_test = pd.read_csv("hwk1_datasets/Datasets/Dataset_2_test.csv")

    Dataset_2_train = np.array(Dataset_2_train)
    Dataset_2_valid = np.array(Dataset_2_valid)
    Dataset_2_test = np.array(Dataset_2_test)


    X_Dataset_2_train = Dataset_2_train [:,0]
    y_Dataset_2_train = Dataset_2_train [:,1]

    X_Dataset_2_valid = Dataset_2_valid[:, 0]
    y_Dataset_2_valid = Dataset_2_valid[:, 1]

    X_Dataset_2_test = Dataset_2_test[:, 0]
    y_Dataset_2_test = Dataset_2_test[:, 1]

    X_train = np.array( X_Dataset_1_train.reshape(len(X_Dataset_1_train),1) )
    y_train = np.array( y_Dataset_1_train.reshape(len(y_Dataset_1_train),1) )

    X_valid = np.array( X_Dataset_1_valid.reshape(len(X_Dataset_1_valid), 1) )
    y_valid = np.array( y_Dataset_1_valid.reshape(len(y_Dataset_1_valid), 1) )

    X_test = np.array( X_Dataset_1_test.reshape(len(X_Dataset_1_test), 1) )
    y_test = np.array( y_Dataset_1_test.reshape(len(y_Dataset_1_test), 1) )

    X_train = add_poly_features(X_train,reg_model_poly_degree)
    Ones = np.ones((len(X_train[:,0]),1))
    X_train = np.concatenate( (Ones,X_train), axis=1)

    X_valid = add_poly_features(X_valid,reg_model_poly_degree)
    Ones = np.ones((len(X_valid[:,0]),1))
    X_valid = np.concatenate( (Ones,X_valid), axis=1)

    X_test = add_poly_features(X_test,reg_model_poly_degree)
    Ones = np.ones((len(X_test[:,0]),1))
    X_test = np.concatenate( (Ones,X_test), axis=1)

    W = np.random.random((2,1))*10
    MSE_for_each_epoch_train = []
    MSE_for_each_epoch_valid = []
    epochs = []
    MSE_for_each_epoch_train, MSE_for_each_epoch_valid, epochs = stocastic_gradient_descent(W, 10**(-6), y_valid = y_valid, y_train = y_train, X_valid=X_valid, X_train= X_train)

    plt.scatter(epochs, MSE_for_each_epoch_train, marker='o', color = 'r')
    plt.scatter(epochs, MSE_for_each_epoch_valid, marker='o', color = 'b')
    plt.show()

def stocastic_gradient_descent (W, alpha, y_valid, y_train, X_valid, X_train):

    new_W = np.array(W + 0.1)

    MSE_for_each_epoch_train = []
    MSE_for_each_epoch_valid = []
    epochs = []
    ind = 0

    while ((abs(new_W - W)>0).any() ):
        ind += 1
        new_W = np.array(W)
        #new_W = W[1]
        for i in range(0, X_train[:,1].size):
            W[0,0] = np.array(W[0,0] - alpha * (W[0,0] + W[1,0] * X_train[i,1] - y_train[i,0]) )
            W[1,0] = np.array(W[1,0] - alpha * (W[0,0] + W[1,0] * X_train[i,1] - y_train[i,0]) * X_train[i,1])

        y_star_train = model(X_train, W)
        y_star_valid = model(X_valid, W)


        MSE_for_each_epoch_train.append( mean_squared_error(y_train, y_star_train) )
        MSE_for_each_epoch_valid.append( mean_squared_error(y_valid, y_star_valid) )
        epochs.append (ind)


        print (abs(new_W - W)>0.01)
        print (new_W)
        print(W)

    return MSE_for_each_epoch_train, MSE_for_each_epoch_valid, epochs




if __name__ == '__main__':
     main()