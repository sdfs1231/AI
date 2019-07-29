import numpy as np
import matplotlib.pyplot as plt
import h5py
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()

def compute_cost_with_regularization(A,Y,parameters,lambd):
	W=[]
	m=Y.shape[1]
	L=len(parameters)//2
	temp=0
	for l in range(1,L+1):
		W.append(parameters['W'+str(l)])
		temp=np.sum(temp+np.sum(np.square(W[l-1])))
	r_cost=lambd*temp/(2*m)
	c_cost=compute_cost(A,Y)
	cost=r_cost+c_cost

	return cost

# A3, Y_assess, parameters = compute_cost_with_regularization_test_case()

# print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))
# exit()

def backward_propagation_with_regularization(X,Y,cache,lambd):
	m=X.shape[1]
	(Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
	dZ3=A3-Y
	dW3=1. / m * np.dot(dZ3, A2.T) + (lambd * W3) / m
	db3=1. / m * np.sum(dZ3, axis=1, keepdims=True)

	dA2=np.dot(W3.T,dZ3)
	dZ2=np.multiply(dA2,np.int64(A2>0))
	dW2=1. / m * np.dot(dZ2, A1.T) + (lambd * W2) / m
	db2=1. / m * np.sum(dZ2, axis=1, keepdims=True)

	dA1=np.dot(W2.T,dZ2)
	dZ1 = np.multiply(dA1, np.int64(A1 > 0))
	dW1 = 1. / m * np.dot(dZ1, X.T) + (lambd * W1) / m
	db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

	gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
	return gradients

# X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()

# grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd=0.7)
# print ("dW1 = " + str(grads["dW1"]))
# print ("dW2 = " + str(grads["dW2"]))
# print ("dW3 = " + str(grads["dW3"]))
# exit()

def forward_propagation_with_droupout(X,parameters,keep_pro):
	np.random.seed(1)
	W1=parameters['W1']
	b1=parameters['b1']
	W2=parameters['W2']
	b2=parameters['b2']
	W3=parameters['W3']
	b3=parameters['b3']

	Z1=np.dot(W1,X)+b1
	A1=relu(Z1)

	D1=np.random.rand(A1.shape[0],A1.shape[1])
	D1=D1<keep_pro
	A1=A1*D1
	A1=A1/keep_pro

	Z2 = np.dot(W2, A1) + b2
	A2 = relu(Z2)

	D2 = np.random.rand(A2.shape[0], A2.shape[1])     
	D2 = D2 < keep_pro                                             
	A2 = A2 * D2                                  
	A2 = A2 / keep_pro               

	Z3 = np.dot(W3, A2) + b3
	A3 = sigmoid(Z3)
    
	cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
	return A3, cache

# X_assess, parameters = forward_propagation_with_dropout_test_case()

# A3, cache = forward_propagation_with_droupout(X_assess, parameters, keep_pro=0.7)
# print ("A3 = " + str(A3))
# exit()

def backward_propagation_with_droupout(X,Y,cache,keep_pro):
	m=X.shape[1]
	(Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
	dZ3=A3-Y
	dW3=1. /m*np.dot(dZ3,A2.T)
	db3=1. /m*np.sum(dZ3,axis=1,keepdims=True)
	dA2=np.dot(W3.T,dZ3)

	dA2=dA2*D2
	dA2=dA2/keep_pro

	dZ2=np.multiply(dA2,np.int64(A2>0))
	dW2=1. / m * np.dot(dZ2, A1.T)
	db2=1. / m * np.sum(dZ2, axis=1, keepdims=True)
	dA1=np.dot(W2.T, dZ2)

	dA1=dA1 * D1
	dA1=dA1 / keep_pro

	dZ1=np.multiply(dA1, np.int64(A1 > 0))
	dW1=1. / m * np.dot(dZ1, X.T)
	db1=1. / m * np.sum(dZ1, axis=1, keepdims=True)

	gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
	return gradients

# X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()

# gradients = backward_propagation_with_droupout(X_assess, Y_assess, cache, keep_pro=0.8)

# print ("dA1 = " + str(gradients["dA1"]))
# print ("dA2 = " + str(gradients["dA2"]))
# exit()



def model(X,Y,num_iter=30000,learning_rate=0.3,lambd=0,keep_pro=1,print_cost=True):
	costs=[]
	m=X.shape[1]
	layers=[X.shape[0],20,3,1]
	parameters=initialize_parameters(layers)

	for i in range(num_iter):
		if keep_pro==1:
			a3,cache=forward_propagation(X,parameters)
		elif keep_pro<1:
			a3,cache=forward_propagation_with_droupout(X,parameters,keep_pro)
		if lambd==0:
			cost=compute_cost(a3,Y)
		elif lambd!=0:
			cost=compute_cost_with_regularization(a3,Y,parameters,lambd)

		assert(lambd==0 or keep_pro==1)

		if lambd==0 and keep_pro==1:
			grads=backward_propagation(X,Y,cache)
		elif lambd!=0:
			grads=backward_propagation_with_regularization(X,Y,cache,lambd)
		elif keep_pro<1:
			grads=backward_propagation_with_droupout(X,Y,cache,keep_pro)

		parameters=update_parameters(parameters,grads,learning_rate)

		if print_cost and i%10000==0:
			print('Cost after iteration{}:{}'.format(i,cost))
		if print_cost and i%1000==0:
			costs.append(cost)

	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('iterations (x1,000)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
	
	return parameters

# parameters = model(train_X, train_Y)
# print("On the training set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

# plt.title("Model without regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75, 0.40])
# axes.set_ylim([-0.75, 0.65])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

# parameters = model(train_X, train_Y, lambd=0.7)
# print("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

# plt.title("Model with L2-regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

parameters = model(train_X, train_Y, keep_pro=0.86, learning_rate=0.3)

print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())