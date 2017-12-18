# This code is inspired from : https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/

import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_derivative(output):
    return output*(1-output)


# training dataset generation
int2binary = {} # a look up table that matches each integer number to its binary
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits( np.array([range(largest_number)], dtype=np.uint8).T,axis=1)
print binary[1] 
for i in range(largest_number):
    int2binary[i] = binary[i]
#print int2binary

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 32
output_dim = 1


# initialize neural network weights
# generate some random weight with mean=0 and range =[-1, 1]
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1  # weights for connecting a hidden layer to input layer
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 # weights for connecting the hidden layer to the output layer
# this synapse connect hidden units of the hidden layer to itself.
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # weights for connecting the hidden layer to the hidden layer

#updates that happen for synaps themselves due to the gradient descent algorithm.
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
#iterate over 10K samples.
for j in range(10000):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store the result predicted by LSTM in (binary encoded) d.
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list() # store the layer2 derivatives at each time-step
    layer_1_values = list() # keep the layer 1 values
    layer_1_values.append(np.zeros(hidden_dim)) # since layer_1 has no previous hidden layer we initialize it at zero.
    
    # moving forward over different layers.
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer  = sigmoid(input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # output layer (new binary representation)
        # we need to multiply laye_1's output to the (synopsys_1) and
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # decode predicted binary output for LSTM so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        # note that LSTM uses inputs, synaps_0 weights and the previous layer_values to calculate the next.
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim)

    # back-propagate procedure.
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    
    # we need to update our weights with respect to learning rate.
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"

        
