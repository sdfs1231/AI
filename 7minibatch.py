import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def upgrade_parameters_gd(parameters,grads,learning_rate):
	L=len(parameters)//2
	for l in range(L):
		parameters['W'+str(l+1)]=parameters['W'+str(l+1)]-learning_rate*grads['dW'+str(l+1)]
		parameters['b'+str(l+1)]=parameters['b'+str(l+1)]-learning_rate*grads['db'+str(l+1)]
	return parameters

# parameters, grads, learning_rate = update_parameters_with_gd_test_case()

# parameters = upgrade_parameters_gd(parameters, grads, learning_rate)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def random_mini_batches(X,Y,mini_batch_size=64,seed=0):
	np.random.seed(seed)
	m=X.shape[1]
	mini_batches=[]
	number_mini_batches=math.floor(m/mini_batch_size)
	
	permutation=list(np.random.permutation(m))
	X=X[:,permutation]#in order to permutate the X and Y in the order of permutation index
	Y=Y[:,permutation].reshape((1,m))
	for i in range(number_mini_batches):
		shuttle_X=X[:,i*mini_batch_size:(i+1)*mini_batch_size]
		shuttle_Y=Y[:,i*mini_batch_size:(i+1)*mini_batch_size]
		mini_batch=(shuttle_X,shuttle_Y)
		mini_batches.append(mini_batch)
	if m%mini_batch_size!=0:
		shuttle_X=X[:,number_mini_batches*mini_batch_size:]
		shuttle_Y=Y[:,number_mini_batches*mini_batch_size:]
		mini_batch=(shuttle_X,shuttle_Y)
		mini_batches.append(mini_batch)
	return mini_batches

X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

print("第一个mini_batch_X的维度: " + str(mini_batches[0][0].shape))
print("第二个mini_batch_X的维度: " + str(mini_batches[1][0].shape))
print("第三个mini_batch_X的维度: " + str(mini_batches[2][0].shape))
print("第一个mini_batch_Y的维度: " + str(mini_batches[0][1].shape))
print("第二个mini_batch_Y的维度: " + str(mini_batches[1][1].shape)) 
print("第三个mini_batch_Y的维度: " + str(mini_batches[2][1].shape))