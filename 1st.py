import numpy as np
import h5py
import matplotlib.pyplot as plt

def loaddata():
	train_f=h5py.File('datasets/train_catvnoncat.h5','r')
	train_x=np.array(train_f['train_set_x'][:])
	train_y=np.array(train_f['train_set_y'][:])

	train_x_flattern=train_x.reshape(train_x.shape[0],-1).T
	train_x=train_x_flattern/255.
	train_y=train_y.reshape((1,train_y.shape[0]))

	test_f=h5py.File('datasets/test_catvnoncat.h5','r')
	test_x=np.array(test_f['test_set_x'][:])
	test_y=np.array(test_f['test_set_y'][:])

	test_x_flattern=test_x.reshape(test_x.shape[0],-1).T
	test_x=test_x_flattern/255.
	test_y=test_y.reshape((1,test_y.shape[0]))

	classes=np.array(test_f['list_classes'][:])
	return train_x,train_y,test_x,test_y,classes

train_x,train_y,test_x,test_y,classes=loaddata()
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape,classes)
# index=30
# print(train_x[index])
#print('lable: '+str(train_y[:,index])+', its a '+classes[np.squeeze(train_y[:,index])].decode('utf-8')+' image')
#print(train_x_flattern.shape,test_x_flattern.shape)

def init(X,):
	w=np.zeros((X.shape[0],1))
	b=0
	return w,b

def sigmoid(w,X,b):
	z=np.dot(w.T,X)+b
	A=1/(1+np.exp(-z))
	return A

def progate(w,X,b,Y):
	A=sigmoid(w,X,b)
	m=X.shape[1]
	dz=A-Y
	dw=np.dot(X,dz.T)/m
	db=np.sum(dz)/m
 
	grads={'dw':dw,
			'db':db}
	cost= -np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m
	return cost,grads

def optimize(w,X,b,Y,num_iter,learning_rate,print_cost=False):
	costs=[]
	for i in range(num_iter):
		cost,grads=progate(w,X,b,Y)
		dw=grads['dw']
		db=grads['db']

		w=w-learning_rate*dw
		b=b-learning_rate*db

		if i%100==0:
			costs.append(cost)
			if print_cost:
				print('After %s optimize,the cost is %s'%(i,cost))
	params={'w':w,
			'b':b}
	return params,costs

def predict(w,X,b):
	m=X.shape[1]
	Y_pred=np.zeros((1,m))
	A=sigmoid(w,X,b)

	for i in range(A.shape[1]):
		if A[0,i]>=0.5:
			Y_pred[0,i]=1
	return Y_pred

def Model(train_X,train_Y,test_X,test_Y,num_iter,learning_rate,print_cost=False):
	w,b=init(train_X)
	params,costs=optimize(w,train_X,b,train_Y,num_iter,learning_rate,print_cost)
	w=params['w']
	b=params['b']

	Y_pred_train=predict(w,train_X,b)
	Y_pred_test=predict(w,test_X,b)

	print('the train image success rate: {}%'.format(100-np.mean(np.abs(Y_pred_train-train_Y))*100))
	print('the test image success rate: {}%'.format(100-np.mean(np.abs(Y_pred_test-test_Y))*100))

	results={'costs':costs,
			'Y_pred_train':Y_pred_train,
			'Y_pred_test':Y_pred_test,
			'w':w,
			'b':b,
			'learning_rate':learning_rate,
			'num_iteration':num_iter}
	return results

d=Model(train_x,train_y,test_x,test_y,num_iter=2000,learning_rate=0.005,print_cost=True)

index=10
plt.imshow(test_x[:,index].reshape((64,64,3)))
print('lable:'+str(test_y[0,index]+'prediction is '+str(int(d['Y_pred_test'][0,index]))))