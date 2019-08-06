import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# index = 0
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train_flat=X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flat=X_test_orig.reshape(X_test_orig.shape[0],-1).T

X_train=X_train_flat/255.
X_test=X_test_flat/255.

Y_train=convert_to_one_hot(Y_train_orig,6)
Y_test=convert_to_one_hot(Y_test_orig,6)

# print("number of training examples = " + str(X_train.shape[1]))
# print("number of test examples = " + str(X_test.shape[1]))
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(Y_train.shape))
# print("X_test shape: " + str(X_test.shape))
# print("Y_test shape: " + str(Y_test.shape))

def create_placeholders(n_x,n_y):
	X=tf.placeholder(tf.float32,[n_x,None],name='X')
	Y=tf.placeholder(tf.float32,[n_y,None],name='Y')
	
	return X,Y
	

# X, Y = create_placeholders(12288, 6)
# print("X = " + str(X))
# print("Y = " + str(Y))

def init_parameters():
	tf.set_random_seed(1)
	
	W1=tf.get_variable('W1',[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
	b1=tf.get_variable('b1',[25,1],initializer=tf.zeros_initializer())
	
	W2=tf.get_variable('W2',[12,25],initializer=tf.contrib.layers.xavier_initializer(seed=1))
	b2=tf.get_variable('b2',[12,1],initializer=tf.zeros_initializer())
	
	W3=tf.get_variable('W3',[6,12],initializer=tf.contrib.layers.xavier_initializer(seed=1))
	b3=tf.get_variable('b3',[6,1],initializer=tf.zeros_initializer())
	

	parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
	
	return parameters

# tf.reset_default_graph()
# with tf.Session() as sess:
    # parameters = init_parameters()
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))


def forward_propagation(X,parameters):
	L=len(parameters)//2
	temp={}
	temp['A0']=X
	for i in range(0,L):
		temp['Z'+str(i+1)]=tf.add(tf.matmul(parameters['W'+str(i+1)],temp['A'+str(i)]),parameters['b'+str(i+1)])
		if i!=L-1:
			temp['A'+str(i+1)]=tf.nn.relu(temp['Z'+str(i+1)])
	# print(temp.keys())
	return temp['Z'+str(L)]

# tf.reset_default_graph()
# with tf.Session() as sess:
	# X,Y=create_placeholders(12288,6)
	# parameters=init_parameters()
	# Z3=forward_propagation(X,parameters)
	# print(str(Z3))

def compute_cost(Z3,Y):
	logits=tf.transpose(Z3)
	labels=tf.transpose(Y)
	
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
	
	return cost

# tf.reset_default_graph()
# with tf.Session() as sess:
	# X,Y=create_placeholders(12288,6)
	# parameters=init_parameters()
	# Z3=forward_propagation(X,parameters)
	# cost=compute_cost(Z3,Y)
	# print(str(cost))
# exit()

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
			num_epochs = 1500, minibatch_size = 32, print_cost = True):
	ops.reset_default_graph()
	tf.set_random_seed(1)
	seed=3
	(n_x,m)=X_train.shape
	n_y=Y_train.shape[0]
	costs=[]
	
	X,Y=create_placeholders(n_x,n_y)
	
	parameters=init_parameters()
	
	Z3=forward_propagation(X,parameters)
	
	cost=compute_cost(Z3,Y)
	
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	init=tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(init)
		
		for epoch in range(num_epochs):
			epoch_cost=0.
			num_minibatches=int(m/minibatch_size)
			
			seed=seed+1
			
			minibatches=random_mini_batches(X_train,Y_train,minibatch_size,seed)
			
			for minibatch in minibatches:
				(minibatch_X,minibatch_Y)=minibatch
				_,minibatch_cost=sess.run([optimizer,cost],feed_dict={X:minibatch_X, Y:minibatch_Y})
				# _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
				epoch_cost+=minibatch_cost/num_minibatches
			if print_cost and epoch%100==0:
				print('Cost after epoch %i : %f'%(epoch,epoch_cost))
			if print_cost and epoch%5==0:
				costs.append(epoch_cost)
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()
		
		parameters=sess.run(parameters)
		print('Parameter has been trained!')
		
		correct_prediction=tf.equal(tf.argmax(Z3),tf.argmax(Y))
		accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
		
		print('Train Accuracy :',accuracy.eval({X:X_train,Y:Y_train}))
		print('Test Accuracy :',accuracy.eval({X:X_test,Y:Y_test}))
		
		return parameters
	
parameters = model(X_train, Y_train, X_test, Y_test)

import scipy
from PIL import Image
from scipy import ndimage

my_image = "thumbs_up.jpg"

fname = "images/" + my_image
image = np.array(plt.imread(fname))
my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))





