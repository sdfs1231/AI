import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

# def forward_propagation(x,theta):
	# J=np.dot(theta,x)
	# return J

# x,theta=2,4
# J=forward_propagation(x,theta)
# print(J)

# def backward_propagation(x,theta):
	# dthta=x
	# return dthta

# dthta=backward_propagation(x,theta)
# #print(dthta)

# def grad_check(x,theta,epsilon=1e-7):
	# thetaplus=theta+epsilon
	# thetaminus=theta-epsilon
	# J_plus=forward_propagation(x,thetaplus)
	# J_minus=forward_propagation(x,thetaminus)
	
	# gradapprox=(J_plus-J_minus)/(2*epsilon)
	# grad=backward_propagation(x,theta)
	
	# numerator=np.linalg.norm(grad-gradapprox)
	# denominator=np.linalg.norm(grad)+np.linalg.norm(gradapprox)
	# difference=numerator/denominator
	
	# if difference<1e-7:
		# print('backward_propagation correct!')
	# else:
		# print('back_propagation error!')
	# return difference

# difference=grad_check(x,theta)
# #print(difference)


def forward_propagation_n(X,Y,parameters):
	m=X.shape[1]
	W1=parameters['W1']
	b1=parameters['b1']
	W2=parameters['W2']
	b2=parameters['b2']
	W3=parameters['W3']
	b3=parameters['b3']
	
	Z1=np.dot(W1,X)+b1
	A1=relu(Z1)
	
	Z2=np.dot(W2,A1)+b2
	A2=relu(Z2)
	
	Z3=np.dot(W3,A2)+b3
	A3=sigmoid(Z3)
	
	logprobs=np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
	cost=1./m*np.sum(logprobs)
	
	cache=(Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
	
	return cost,cache

def back_propagation_n(X,Y,cache):
	m=X.shape[1]
	(Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
	dZ3=A3-Y
	dW3=1. / m * np.dot(dZ3, A2.T)
	db3=1. / m * np.sum(dZ3, axis=1, keepdims=True)
	
	dA2=np.dot(W3.T, dZ3)
	dZ2=np.multiply(dA2, np.int64(A2 > 0))
	dW2=1. / m * np.dot(dZ2, A1.T) 
	db2=1. / m * np.sum(dZ2, axis=1, keepdims=True)
	
	dA1=np.dot(W2.T, dZ2)
	dZ1=np.multiply(dA1, np.int64(A1 > 0))
	dW1=1. / m * np.dot(dZ1, X.T)
	db1=1. / m * np.sum(dZ1, axis=1, keepdims=True)
	gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
	return gradients

def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
	parameters_values,_=dictionary_to_vector(parameters)
	grad=gradients_to_vector(gradients)
	num_parameters=parameters_values.shape[0]
	J_plus=np.zeros((num_parameters,1))
	J_minus=np.zeros((num_parameters,1))
	gradapprox=np.zeros((num_parameters,1))
	
	for i in range(num_parameters):
		thetaplus=np.copy(parameters_values)
		thetaplus[i][0]=thetaplus[i][0]+epsilon
		J_plus[i],_=forward_propagation_n(X,Y,vector_to_dictionary(thetaplus))
		
		thetaminus=np.copy(parameters_values)
		thetaminus[i][0]=thetaminus[i][0]-epsilon
		J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))
		gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
	numerator=np.linalg.norm(grad-gradapprox)
	denominator=np.linalg.norm(grad)+np.linalg.norm(gradapprox)
	difference=numerator/denominator
	
	if difference>2e-7:
		print("\033[93m" + "反向传播有问题! difference = " + str(difference) + "\033[0m")
	else:
		print("\033[92m" + "反向传播很完美! difference = " + str(difference) + "\033[0m")
	return difference
	
X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = back_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)