import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt


def add_poly_features(X, poly_degree):
    if (poly_degree>=2):
        X = np.concatenate(( X, (X[:,0] ** 2).reshape(len(X[:,0]),1) ), axis=1)
        #print(X)
        for i in range(3, poly_degree+1):
            X = np.concatenate((X, (X[:,0] ** i).reshape(len(X[:,0]),1)), axis=1)
            #print(X)

    return X


def main ():

    reg_model_poly_degree = 10
    lambd = 0.002

    Dataset_1_train = pd.read_csv("hwk1_datasets/Datasets/Dataset_1_train.csv")
    Dataset_1_test = pd.read_csv("hwk1_datasets/Datasets/Dataset_1_test.csv")
    Dataset_1_valid = pd.read_csv("hwk1_datasets/Datasets/Dataset_1_valid.csv")

    Dataset_1_train = np.array(Dataset_1_train)
    Dataset_1_test = np.array(Dataset_1_test)
    Dataset_1_valid = np.array(Dataset_1_valid)



    X_Dataset_1_train = np.array( Dataset_1_train [:,0] )
    y_Dataset_1_train = np.array( Dataset_1_train [:,1] )

    X_Dataset_1_valid = np.array( Dataset_1_valid[:, 0] )
    y_Dataset_1_valid = np.array( Dataset_1_valid[:, 1] )

    X_Dataset_1_test = np.array( Dataset_1_test[:, 0] )
    y_Dataset_1_test = np.array( Dataset_1_test[:, 1] )
    #
    # plt.scatter(X_Dataset_1_train, y_Dataset_1_train, marker='o', edgecolors='r')
    # plt.scatter(X_Dataset_1_valid, y_Dataset_1_valid, marker='o', edgecolors='b')
    # plt.show()


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



    W_star_train = W_star(X_train,y_train)


    y_star_train = model(X_train, W_star_train)
    RSS_W_start_train = mean_squared_error(y_train, y_star_train)
    print("tarining set's MSE is:" + str(RSS_W_start_train))

    y_star_valid = model(X_valid, W_star_train)
    RSS_W_start_valid = mean_squared_error(y_valid, y_star_valid)
    print("validation set's MSE is:" + str(RSS_W_start_valid))

    y_star_test = model(X_test, W_star_train)
    RSS_W_start_test = mean_squared_error(y_test, y_star_test)
    print("test set's MSE is:" + str(RSS_W_start_test))

    #plt.scatter(X_Dataset_1_train, y_star_train, marker='o', linestyle='--', color='r', label='Approximate_Model')
    #plt.scatter(X_Dataset_1_train, y_train, marker='o', linestyle='--', color='b', label='Train_Data')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.title('model approximation over training dataset')
    #plt.legend()
    #plt.show()

    model_MSE_over_train_with_Regu = mean_squared_error_with_regu(y_train, y_star_train, lambd, W_star_train)

    print("tarining set's model_MSE_over_train_with_Regu is:" + str(model_MSE_over_train_with_Regu))

    train_MSE_for_diff_Lambda = [x for x in np.arange(0,1,0.001)]
    valid_MSE_for_diff_Lambda = [x for x in np.arange(0,1,0.001)]
    ind = 0

    # Sweep lambda for regularization
    lambda_range = np.arange(0,1,0.001)


    for i in lambda_range:
        train_MSE_for_diff_Lambda[ind] = mean_squared_error_with_regu(y_train, y_star_train, i, W_star_train)
        ind +=1

    ind = 0
    for i in lambda_range:
        valid_MSE_for_diff_Lambda[ind] = mean_squared_error_with_regu(y_valid, y_star_valid, i, W_star_train)
        ind +=1
    #print(i)
    print(valid_MSE_for_diff_Lambda)
    plt.scatter(lambda_range, valid_MSE_for_diff_Lambda, marker='o', color='r')
    plt.show()


def W_star(X, y):
    return np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)

def model (X, W_star):
    return np.dot(X, W_star)

def mean_squared_error (y, y_star):
     temp=y - y_star
     temp2 = np.dot(temp.T, temp)
     temp3  = (1.0/2.0)*(1.0/len(y[:,0])) * temp2
     return (1.0/2.0)*(1.0/len(y[:,0])) * np.dot( (y-y_star).T, (y-y_star) )


def mean_squared_error_with_regu(y, y_star, lambd , W_star):
    return (1.0/2.0)*(1.0 / len(y[:, 0])) * np.dot((y - y_star).T, (y - y_star)) + (1/2)*lambd*np.dot(W_star.T, W_star)


if __name__ == '__main__':
    main()










