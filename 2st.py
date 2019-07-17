import numpy as np
import sklearn
import matplotlib.pyplot as plt
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets 
from testCases import * 

X,Y=load_planar_dataset()
#print(X.shape,Y.shape)

clf=sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T.ravel())
LR_predictions=clf.predict(X.T)
# print(np.dot(Y,LR_predictions),np.dot(1-Y,1-LR_predictions))

#plot_decision_boundary(lambda x:clf.predict(x),X,Y.ravel())

def init_params(n_x,n_h,n_y):

	np.random.seed(2)
	W1=np.random.randn(n_x,n_h)*0.01
	b1=np.zeros((n_h,1))
	W2=np.random.randn(n_h,n_y)*0.01
	b2=np.zeros((n_y,1))
	parameters={
				'W1':W1,
				'b1':b1,
				'W2':W2,
				'b2':b2
	}
	return parameters

# n_x,n_h,n_y=initialize_parameters_test_case()
# parameters=init_params(n_x,n_h,n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# print("--------------------------------------------")

# print("nn dim(" + str(n_x) + "," + str(n_h) + "," + str(n_y) + ")");
# print("W1 dim:" + str(parameters["W1"].shape))
# print("b1 dim:" + str(parameters["b1"].shape))
# print("W2 dim:" + str(parameters["W2"].shape))
# print("b2 dim:" + str(parameters["b2"].shape))

def forward_pro(X,parameters):
	b1=parameters['b1']
	W1=parameters['W1']
	Z1=np.dot(W1.T,X)+b1
	A1=np.tanh(Z1)

	b2=parameters['b2']
	W2=parameters['W2']
	Z2=np.dot(W2.T,A1)+b2
	A2=sigmoid(Z2)

	cache={
			'Z1':Z1,
			'A1':A1,
			'Z2':Z2,
			'A2':A2
	}

	return A2,cache

# X_assess, parameters = forward_propagation_test_case()

# A2, cache = forward_pro(X_assess, parameters)

# print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

# print("--------------------------------------------")

# print("W1 dim:" + str(parameters["W1"].shape))
# print("b1 dim:" + str(parameters["b1"].shape))
# print("W2 dim:" + str(parameters["W2"].shape))
# print("b2 dim:" + str(parameters["b2"].shape))

# print("--------------------------------------------")

# print("Z1 dim:" + str(cache['Z1'].shape))
# print("A1 dim:" + str(cache['A1'].shape))
# print("Z2 dim:" + str(cache['Z2'].shape))
# print("A2 dim:" + str(cache['A2'].shape))

def compute_cost(A2,Y):
	m = Y.shape[1]
	logprobs = np.multiply(Y,np.log(A2))+np.multiply((1-Y),np.log(1-A2))
	cost = - np.sum(logprobs) / m
	return cost

# A2, Y_assess = compute_cost_test_case()

# print("cost = " + str(compute_cost(A2, Y_assess)))

def back_pro(parameters,cache,X,Y):
	m=X.shape[1]
	W1=parameters['W1']
	A1=cache['A1']

	W2=parameters['W2']
	A2=cache['A2']

	dZ2=A2-Y
	dW2=np.dot(dZ2,A1.T)/m
	db2=np.sum(dZ2,axis=1,keepdims=True)/m

	dZ1=np.multiply(np.dot(W2,dZ2),1-np.power(A1,2))
	dW1=np.dot(dZ1,X.T)/m
	db1=np.sum(dZ1,axis=1,keepdims=True)/m

	dW1=dW1.T
	dW2=dW2.T
	grads={'dW1':dW1,
			'db1':db1,
			'dW2':dW2,
			'db2':db2}

	return grads

# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

# grads = back_pro(parameters, cache, X_assess, Y_assess)

# print("dW1 dim:" + str(grads["dW1"].shape))
# print("db1 dim:" + str(grads["db1"].shape))
# print("dW2 dim:" + str(grads["dW2"].shape))
# print("db2 dim:" + str(grads["db2"].shape))
# print("------------------------------")

# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

def update_parameters(parameters,grads,learning_rate=1.2):
	W1=parameters['W1']
	dW1=grads['dW1']
	b1=parameters['b1']
	db1=grads['db1']
	W2=parameters['W2']
	dW2=grads['dW2']
	b2=parameters['b2']
	db2=grads['db2']

	W1=W1-learning_rate*dW1
	b1=b1-learning_rate*db1

	W2=W2-learning_rate*dW2
	b2=b2-learning_rate*db2

	parameters={
				'W1':W1,
				'b1':b1,
				'W2':W2,
				'b2':b2
	}
	return parameters

# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)

# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def model(X,Y,n_h,num_iter=100000,print_cost=False):
	np.random.seed(3)
	n_x=X.shape[0]
	n_y=Y.shape[0]

	parameters=init_params(n_x,n_h,n_y)
	W1=parameters['W1']
	b1=parameters['b1']
	W2=parameters['W2']
	b2=parameters['b2']

	for i in range(0,num_iter):
		A2,cache=forward_pro(X,parameters)
		cost=compute_cost(A2,Y)
		grads=back_pro(parameters,cache,X,Y)
		parameters=update_parameters(parameters,grads)
		if print_cost and i%1000==0:
			print('after %i times traing,the cost is: %f'%(i,cost))
	return parameters

# X_assess, Y_assess = nn_model_test_case()

# parameters = model(X_assess, Y_assess, 4, num_iter=10000, print_cost=False)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def predict(X,parameters):
	A2,cache=forward_pro(X,parameters)
	predictions=np.round(A2)

	return predictions

# parameters, X_assess = predict_test_case()

# predictions = predict(X_assess,parameters)
# print("predictions mean = " + str(np.mean(predictions)))

parameters = model(X, Y, n_h = 4, num_iter=10000, print_cost=True)

# 然后用训练得出的参数进行预测。
predictions = predict(X,parameters)
print ('预测准确率是: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

# 将预测结果画出来。
plot_decision_boundary(lambda x: predict(x.T,parameters), X, Y.ravel())

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50] # 不同的神经元个数
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = model(X, Y, n_h, num_iter=5000)
    plot_decision_boundary(lambda x: predict(x.T,parameters), X, Y.ravel())
    predictions = predict(X,parameters)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print ("{}个隐藏层神经元时的准确度是: {} %".format(n_h, accuracy))

	