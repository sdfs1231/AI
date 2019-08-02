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

def init_velovity(parameters):
	L=len(parameters)//2
	v={}
	for i in range(L):
		v['dW'+str(i+1)]=np.zeros_like(parameters['W'+str(i+1)])
		v['db'+str(i+1)]=np.zeros_like(parameters['b'+str(i+1)])
	return v
	
# parameters = initialize_velocity_test_case()

# v = init_velovity(parameters)
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))

def update_parameters_with_momentum(parameters,grads,v,beta,learning_rate):
	L=len(parameters)//2
	
	for i in range(L):
		v['dW'+str(i+1)]=beta*v['dW'+str(i+1)]+(1-beta)*grads['dW'+str(i+1)]
		v['db'+str(i+1)]=beta*v['db'+str(i+1)]+(1-beta)*grads['db'+str(i+1)]
		
		parameters['W'+str(i+1)]=parameters['W'+str(i+1)]-learning_rate*v['dW'+str(i+1)]
		parameters['b'+str(i+1)]=parameters['b'+str(i+1)]-learning_rate*v['db'+str(i+1)]
	return parameters,v
# 这里的指数加权平均值是没有添加修正算法的。所以在前面一小段的梯度下降中，趋势平均值是不准确的。
# 如果𝛽b=0
# ,那么上面的就成了一个普通的标准梯度下降算法了。
# b越大，那么学习路径就越平滑，因为与指数加权平均值关系紧密的梯度值就越多。但是，如果b
# 太大了，那么它就不能准确地实时反应出梯度的真实情况了.
# 一般来说，𝛽b
# 的取值范围是0.8到0.999。𝛽=0.9
# 是最常用的默认值。
# 当然，你可以尝试0.9之外的值，也许能找到一个更合适的值。建议大家尝试尝试。
# parameters, grads, v = update_parameters_with_momentum_test_case()

# parameters, v = update_parameters_with_momentum(parameters, grads, v, beta =0.9, learning_rate = 0.01)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))