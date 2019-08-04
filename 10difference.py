import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_grad(parameters,grads,learning_rate):
	L=len(parameters)//2

	for i in range(L):
		parameters['W'+str(i+1)]=parameters['W'+str(i+1)]-learning_rate*grads['dW'+str(i+1)]
		parameters['b'+str(i+1)]=parameters['b'+str(i+1)]-learning_rate*grads['db'+str(i+1)]
	return parameters

def random_mini_batches(X,Y,mini_batch_size=64,seed=0):
	np.random.seed(seed)
	m=X.shape[1]
	mini_batches=[]
	permutation=list(np.random.permutation(m))

	shuffle_X=X[:,permutation]
	shuffle_Y=Y[:,permutation].reshape((1,m))

	num=math.floor(m/mini_batch_size)

	for i in range(num):
		mini_X=shuffle_X[:,i*mini_batch_size:(i+1)*mini_batch_size]
		mini_Y=shuffle_Y[:,i*mini_batch_size:(i+1)*mini_batch_size]

		mini_batch=(mini_X,mini_Y)
		mini_batches.append(mini_batch)

	if m%mini_batch_size!=0:
		mini_X=shuffle_X[:,num*mini_batch_size:]
		mini_Y=shuffle_Y[:,num*mini_batch_size:]
		mini_batch=(mini_X,mini_Y)
		mini_batches.append(mini_batch)
	return mini_batches

def init_velocity(parameters):
	L=len(parameters)//2
	v={}

	for i in range(L):
		v['dW'+str(i+1)]=np.zeros_like(parameters['W'+str(i)])
		v['db'+str(i+1)]=np.zeros_like(parameters['b'+str(i)])
	return v

def update_parameters_with_momentum(parameters,grads,v,beta,learning_rate):
	L=len(parameters)//2

	for i in range(L):
		v['dW'+str(i+1)]=beta*v['dW'+str(i+1)]+(1-beta)*grads['dW'+str(i+1)]
		v['db'+str(i+1)]=beta*v['db'+str(i+1)]+(1-beta)*grads['db'+str(i+1)]

		parameters['W'+str(i+1)]=parameters['W'+str(i+1)-learning_rate]*v['dW'+str(i+1)]
		parameters['b'+str(i+1)]=parameters['b'+str(i+1)-learning_rate]*v['db'+str(i+1)]
	return parameters,v

def init_adam(parameters):
	L=len(parameters)//2
	v={}
	s={}

	for i in range(L):
		v['dW'+str(i+1)]=np.zeros_like(parameters['W'+str(i+1)])
		v['db'+str(i+1)]=np.zeros_like(parameters['b'+str(i+1)])


		s['dW'+str(i+1)]=np.zeros_like(parameters['W'+str(i+1)])
		s['db'+str(i+1)]=np.zeros_like(parameters['b'+str(i+1)])

	return v,s

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
	L=len(parameters)//2
	v_corr={}
	s_corr={}

	for i in range(L):
		v['dW'+str(i+1)]=beta1*v['dW'+str(i+1)]+(1-beta1)*grads['dW'+str(i+1)]
		v['db'+str(i+1)]=beta1*v['db'+str(i+1)]+(1-beta1)*grads['db'+str(i+1)]

		v_corr['dW'+str(i+1)]=v['dW'+str(i+1)]/(1-np.power(beta1,t))
		v_corr['db'+str(i+1)]=v['db'+str(i+1)]/(1-np.power(beta1,t))

		s['dW'+str(i+1)]=beta2*s['dW'+str(i+1)]+(1-beta2)*np.power(grads['dW'+str(i+1)],2)
		s['db'+str(i+1)]=beta2*s['db'+str(i+1)]+(1-beta2)*np.power(grads['db'+str(i+1)],2)

		s_corr['dW'+str(i+1)]=s['dW'+str(i+1)]/(1-np.power(beta2,t))
		s_corr['db'+str(i+1)]=s['db'+str(i+1)]/(1-np.power(beta2,t))

		parameters['W'+str(i+1)]=parameters['W'+str(i+1)]-learning_rate*v_corr['dW'+str(i+1)]/np.sqrt(s_corr['dW'+str(i+1)]+epsilon)
		parameters['b'+str(i+1)]=parameters['b'+str(i+1)]-learning_rate*v_corr['db'+str(i+1)]/np.sqrt(s_corr['db'+str(i+1)]+epsilon)
	return parameters,v,s

train_X, train_Y = load_dataset()
#
def model(X,Y,layers,optimizer,learning_rate=0.0007,mini_batch_size=64,beta=0.9,
	beta1=0.9,beta2=0.999,epsilon=1e-8,num_epochs=10000,print_cost=True):
	L=len(layers)

	costs=[]
	t=0
	seed=10
	parameters=initialize_parameters(layers)

	if optimizer=='gd':
		pass
	elif optimizer=='momentum':
		v=init_velocity(parameters)
	elif optimizer=='adam':
		v,s=init_adam(parameters)

	for i in range(num_epochs):
		seed=seed+1
		minibatches=random_mini_batches(X,Y,mini_batch_size,seed)

		for minibatch in minibatches:
			(minibatch_X,minibatch_Y)=minibatch
			# print(minibatch_Y.shape)
			a3,caches=forward_propagation(minibatch_X,parameters)
			cost=compute_cost(a3,minibatch_Y)

			grads=backward_propagation(minibatch_X,minibatch_Y,caches)

			if optimizer=='gd':
				parameters=update_parameters_with_grad(parameters,grads,learning_rate)
			elif optimizer=='momentum':
				parameters=update_parameters_with_momentum(parameters,grads,v,beta,learning_rate)
			elif optimizer=='adam':
				t=t+1
				parameters,v,s=update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
		if print_cost and i%1000==0:
			print('cost after epoch %i:%f'%(i,cost))
		if print_cost and i%100==0:
			costs.append(cost)
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('epochs (per 100)')
	plt.title("Learning rate = " + str(learning_rate))
	plt.show()

	return parameters

# layers_dims = [train_X.shape[0], 5, 2, 1]
# parameters = model(train_X, train_Y, layers_dims, optimizer="gd")

# predictions = predict(train_X, train_Y, parameters)

# plt.title("Model with Gradient Descent optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 2.5])
# axes.set_ylim([-1, 1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer="adam")

predictions = predict(train_X, train_Y, parameters)

plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())