import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import numpy as np
from keras.layers.recurrent import LSTM
import time


def data_generation (numBits=5, numofSamples=1000):

    A = 2**numBits * np.random.random((numofSamples, 1))
    A = A.astype(int)

    B = 2 ** numBits * np.random.random((numofSamples, 1))
    B = B.astype(int)

    C = A + B
    trainInputA  =  A[0:int(numofSamples*0.6), -1]
    trainInputB  =  B[0:int(numofSamples * 0.6), -1]
    trainOutputC =  C[0:int(numofSamples * 0.6), -1]

    validInputA    = A[int(numofSamples * 0.6): int(numofSamples * 0.75), -1]
    validInputB    = B[int(numofSamples * 0.6): int(numofSamples * 0.75), -1]
    validOutputC   = C[int(numofSamples * 0.6): int(numofSamples * 0.75), -1]

    testInputA    = A[int(numofSamples * 0.75): int(numofSamples * 1.), -1]
    testInputB    = B[int(numofSamples * 0.75): int(numofSamples * 1.), -1]
    testOutputC   = C[int(numofSamples * 0.75): int(numofSamples * 1.), -1]

    return trainInputA, trainInputB, trainOutputC, validInputA, validInputB, validOutputC, testInputA, testInputB, testOutputC

def bitfield(n, numBits):
    binaryNum = [int(digit) for digit in bin(n)[2:]] # [2:] to chop off the "0b" part
    temp =  [0]*(numBits)
    temp[numBits - len(binaryNum):] = binaryNum
    return temp

def main():

    numBits = 40
    sequence_length = 1

    trainInputA, trainInputB, trainOutputC, validInputA, validInputB, validOutputC, testInputA, testInputB, testOutputC = data_generation (numBits, 1000)


    model = Sequential()

    model.add(LSTM(
        input_dim=2 * numBits, # it defines the input size
        output_dim=sequence_length, # defines the number of neurons in the first layer of LSTM
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        2*numBits,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense( # fully connected layer
        output_dim=numBits+1)) # it has just one neuron as the output
    model.add(Activation('linear'))

    start = time.time()

    model.compile(loss='mse', optimizer='rmsprop')
    print ('compilation time : ', time.time() - start)

    # for i in range (0, len(trainInputA)):
    #     temp1 = bitfield(trainInputA[i])
    #     temp2 = bitfield(trainInputB[i])

    # y_train = np.zeros((len(trainOutputC), numBits +1 ))
    # print (trainInputB[0], trainInputA[0])
    # print (len (trainInputA))

    # X_train = np.empty((len(trainInputA),2*numBits), dtype=int)
    # y_train = np.empty((len(trainOutputC), numBits+1), dtype=int)

    #X_train[0] = trainInputB[0], trainInputA[0]
    #X_train[0] = np.concatenate((bitfield(trainInputA[0], numBits), bitfield(trainInputB[0], numBits)))

    # y_train =[[]]
    X_train = []
    y_train = []
    for i in range(0, len(trainInputA)):
        X_train.append ( ( bitfield(trainInputA[i], numBits) + bitfield(trainInputB[i], numBits) ) )
        y_train.append( ( bitfield(trainOutputC[i], numBits+1) )  )

    result  =[]
    for index in range(len(X_train) - sequence_length -1):
        result.append(X_train[index: index + sequence_length])

    result = np.array(result)
    # print (len(result) )
    # print (len(result[0]))
    # print (len(result[0][0]))

    # print(X_train.shape[:])

    # y_train = keras.utils.to_categorical([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0], num_classes=None)
    print(y_train[0])
    y = np.array(y_train[0:2])
    # print (y)
    # y = np.reshape(y,(y.shape[0],1))
    # print(y)
#Step 3 Train the model
    model.fit(
        np.array(result),
        np.array(y_train),
        batch_size=16,
        nb_epoch=100,
        validation_split=0.05, verbose=0)



if __name__=='__main__':
    # print (np.concatenate(  ( bitfield(12, 5), bitfield(8, 5) )  ))
    main()
    #bitfield(123, 9)