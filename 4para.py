import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

# 加载自定义的工具库
from init_utils import *

# 设置好画图工具
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_dataset()

def initialize_parameters_zeros(layers):
	parameters={}
	L=len(layers)

	for l in range(1,L):
		parameters['W'+str(l)]=np.zeros((layers[l],layers[l-1]))
		parameters['b'+str(l)]=np.zeros((layers[l],1))
	return parameters
# parameters = initialize_parameters_zeros([3,2,1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
def initialize_parameters_random(layers):
	np.random.seed(3)
	parameters={}
	L=len(layers)

	for l in range(1,L):
		parameters['W'+str(l)]=np.random.randn(layers[l],layers[l-1])*10
		parameters['b'+str(l)]=np.zeros((layers[l],1))
	return parameters

# parameters = initialize_parameters_random([3, 2, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# exit()

def initialize_parameters_he(layers):
	np.random.seed(3)
	parameters={}
	L=len(layers)

	for l in range(1,L):
		parameters['W'+str(l)]=np.random.randn(layers[l],layers[l-1])*np.sqrt(2/layers[l-1])
		parameters['b'+str(l)]=np.zeros((layers[l],1))
	return parameters

# parameters = initialize_parameters_he([2, 4, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# exit()

def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization='he'):
	grads={}
	costs=[]
	m=X.shape[1]
	layers=[X.shape[0],10,5,1]

	if initialization=='zeros':
		parameters=initialize_parameters_zeros(layers)
	elif initialization=='random':
		parameters=initialize_parameters_random(layers)
	elif initialization=='he':
		parameters=initialize_parameters_he(layers)

	for i in range(0,num_iterations):
		a3,cache=forward_propagation(X,parameters)
		cost=compute_loss(a3,Y)
		grads=backward_propagation(X,Y,cache)
		parameters=update_parameters(parameters,grads,learning_rate)

		if print_cost and i%1000==0:
			print('Cost after iteration{}:{}'.format(i,cost))
			costs.append(cost)
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title('learning_rate='+str(learning_rate))
	plt.show()

	return parameters

# parameters = model(train_X, train_Y, initialization = "zeros")
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters) # 对训练数据进行预测，并打印出准确度
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

# parameters = model(train_X, train_Y, initialization = "random")
# print("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

# plt.title("Model with large random initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

parameters = model(train_X, train_Y, initialization = "he")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)