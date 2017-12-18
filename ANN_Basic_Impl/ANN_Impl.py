import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))



x= np.array([[0,1,0],
             [0,1,1],
             [1,0,1],
             [1,1,1]])

y=np.array([[0],
            [1],
            [1],
            [0]])

print (x)

num_epochs = 6000

weights0 = 2 * np.random.random((3,4)) -1
weights1 = 2 * np.random.random((4,1)) -1

print (weights0)
print (weights1)


for i in range(0, num_epochs):
    l0 = x
    l1 = nonlin(np.dot(l0, weights0))
    l2 = nonlin(np.dot(l1, weights1))


    l2_error = y - l2
    mse = ((l2_error) ** 2).mean(axis=0)
    print(mse)


    l2_delta = l2_error * nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(weights1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)

    weights1 += l1.T.dot(l2_delta)
    weights0 += l0.T.dot(l1_delta)
