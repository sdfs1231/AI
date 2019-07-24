import numpy as np
import matplotlib.pyplot as plt
import h5py
from testCases import *
from dnn_utils import *

# 设置一些画图相关的参数
plt.rcParams['figure.figsize'] = (5.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
m_train = train_x_orig.shape[0] # 训练样本的数量
m_test = test_x_orig.shape[0] # 测试样本的数量
num_px = test_x_orig.shape[1] # 每张图片的宽/高

# 为了方便后面进行矩阵运算，我们需要将样本数据进行扁平化和转置
# 处理后的数组各维度的含义是（图片数据，样本数）
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T 

# 下面我们对特征数据进行了简单的标准化处理（除以255，使所有值都在[0，1]范围内）
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

layers_dims = [12288, 20, 7, 5, 1]

def init_wb(layers):
	# A_pre=X
	np.random.seed(1)
	parameters={}
	for i in range(1,len(layers)):
		parameters['W'+str(i)]=np.random.randn(layers[i],layers[i-1])//np.sqrt(layers[i-1])
		parameters['b'+str(i)]=np.zeros((layers[i],1))
	return parameters#[w1,b1,w2,b2....wn,wn],n=layers-1

def linear_forward(X,W,b,activation):
	Z=np.dot(W,X)+b
	assert(Z.shape==(W.shape[0],X.shape[1]))
	if activation=='sigmoid':
		A=sigmoid(Z)
	elif activation=='relu':
		A=relu(Z)
	cache=((X,W,b),Z)
	return A,cache
	#return cache is 2

def model_L_forward(X,parameters):
	A_pre=X
	caches=[]
	L=len(parameters)//2
	for i in range(1,L):
		print(i)
		print(A_pre)
		A,cache=linear_forward(A_pre,
							parameters['W'+str(i)],
							parameters['b'+str(i)],
							activation='relu')
		A_pre=A
		caches.append(cache)#((X,W,b),Z)format
	A,cache=linear_forward(A_pre,
							parameters['W'+str(L)],
							parameters['b'+str(L)],
							activation='sigmoid')
	#up mention A is final A
	assert(A.shape==(1,X.shape[1]))
	caches.append(cache)
	return A,caches
parameters=init_wb(layers_dims)
A,caches=model_L_forward(train_x,parameters)
print(A)
exit()
def compute_cost(A,Y):
	m=Y.shape[1]
	cost=-np.sum(np.multiply(Y,np.log(A))+np.multiply((1-Y),np.log(1-A)))/m
	cost=np.squeeze(cost)
	assert(cost.shape==())
	return cost

def linear_backward(dA,L,cache,activation):
	A_pre,W,b=cache[0]
	Z=cache[1]
	m=L
	if activation=='relu':
		dZ=relu_backward(dA,Z)
	elif activation=='sigmoid':
		dZ=sigmoid_backward(dA,Z)
	dW=np.dot(dZ,A_pre.T)/m
	db=np.sum(dZ,axis=1,keepdims=True)/m
	dA_pre=np.dot(W.T,dZ)
	assert (dA_pre.shape == A_pre.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)

	return dA_pre,dW,db

# dAL, linear_activation_cache = linear_activation_backward_test_case()#cache=((A,W,b),Z)
# L=len(linear_activation_cache)
# dA_prev, dW, db = linear_backward(dAL,L, linear_activation_cache, activation = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")

# dA_prev, dW, db = linear_backward(dAL,L, linear_activation_cache, activation = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

def model_L_backward(AL,Y,caches):
	grads={}
	L=len(caches)#2
	Y=Y.reshape(AL.shape)
	dAL=-(np.divide(Y,AL)-np.divide((1-Y),(1-AL)))
	grads['dA'+str(L-1)],grads['dW'+str(L)],grads['db'+str(L)]=linear_backward(dAL,L,
															caches[L-1],activation='sigmoid')

	for i in reversed(range(1,L)):
		# print(i)
		grads['dA'+str(i-1)],grads['dW'+str(i)],grads['db'+str(i)]=linear_backward(grads['dA'+str(i)],L,
															caches[i-1],activation='relu')
	return grads

# AL, Y_assess, caches = L_model_backward_test_case()
# grads = model_L_backward(AL, Y_assess, caches)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dA1 = "+ str(grads["dA1"]))
# exit()

def update_parameters(parameters,grads,learning_rate):
	L=len(parameters)//2
	for l in range(1,L+1):
		parameters['W'+str(l)]=parameters['W'+str(l)]-learning_rate*grads['dW'+str(l)]
		parameters['b'+str(l)]=parameters['b'+str(l)]-learning_rate*grads['db'+str(l)]
	return parameters
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)

# print ("W1 = " + str(parameters["W1"]))
# print ("b1 = " + str(parameters["b1"]))
# print ("W2 = " + str(parameters["W2"]))
# print ("b2 = " + str(parameters["b2"]))
# exit()

def dnn_model(X,Y,layers,num_iter=3000,learning_rate=0.0075,print_cost=False):
	np.random.seed(1)
	costs=[]
	parameters=init_wb(layers)
	for i in range(0,num_iter):
		AL,caches=model_L_forward(X,parameters)
		cost=compute_cost(AL,Y)
		grads=model_L_backward(AL,Y,caches)
		parameters=update_parameters(parameters,grads,learning_rate)
		if i==0:
			print(AL.shape)
		if i%500==0:
			if print_cost and i>0:
				print('After %i times training,the cost is :%f'%(i,cost))
				costs.append(cost)
				print('AL='+str(AL))
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
	return parameters

# 根据上面的层次信息来构建一个深度神经网络，并且用之前加载的数据集来训练这个神经网络，得出训练后的参数
parameters = dnn_model(train_x, train_y, layers_dims, num_iter=2000, print_cost=True)