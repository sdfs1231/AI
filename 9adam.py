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

def initialize_adam(parameters):
	L=len(parameters)//2
	v={}
	s={}
	for i in range(L):
		v['dW'+str(i+1)]=np.zeros_like(parameters['W'+str(i+1)])
		v['db'+str(i+1)]=np.zeros_like(parameters['b'+str(i+1)])
		s['dW'+str(i+1)]=np.zeros_like(parameters['W'+str(i+1)])
		s['db'+str(i+1)]=np.zeros_like(parameters['b'+str(i+1)])
	
	return v,s
	
# parameters = initialize_adam_test_case()

# v, s = initialize_adam(parameters)
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))
# print("s[\"dW1\"] = " + str(s["dW1"]))
# print("s[\"db1\"] = " + str(s["db1"]))
# print("s[\"dW2\"] = " + str(s["dW2"]))
# print("s[\"db2\"] = " + str(s["db2"]))

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
	L=len(parameters)//2
	v_corrected={}
	s_corrected={}
	
	for i in range(L):
		v['dW'+str(i+1)]=beta1*v['dW'+str(i+1)]+(1-beta1)*grads['dW'+str(i+1)]
		v['db'+str(i+1)]=beta1*v['db'+str(i+1)]+(1-beta1)*grads['db'+str(i+1)]
		
		v_corrected['dW'+str(i+1)]=v['dW'+str(i+1)]/(1-np.power(beta1,t))
		v_corrected['db'+str(i+1)]=v['db'+str(i+1)]/(1-np.power(beta1,t))
		
		s['dW'+str(i+1)]=beta2*s['dW'+str(i+1)]+(1-beta2)*np.power(grads['dW'+str(i+1)],2)
		s['db'+str(i+1)]=beta2*s['db'+str(i+1)]+(1-beta2)*np.power(grads['db'+str(i+1)],2)
		
		s_corrected['dW'+str(i+1)]=s['dW'+str(i+1)]/(1-np.power(beta2,t))
		s_corrected['db'+str(i+1)]=s['db'+str(i+1)]/(1-np.power(beta2,t))
		
		parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * v_corrected["dW" + str(i + 1)] / np.sqrt(s["dW" + str(i + 1)] + epsilon)
		parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * v_corrected["db" + str(i + 1)] / np.sqrt(s["db" + str(i + 1)] + epsilon)
		
	return parameters,v,s

parameters, grads, v, s = update_parameters_with_adam_test_case()
parameters, v, s  = update_parameters_with_adam(parameters, grads, v, s, t = 2)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
print("s[\"dW1\"] = " + str(s["dW1"]))
print("s[\"db1\"] = " + str(s["db1"]))
print("s[\"dW2\"] = " + str(s["dW2"]))
print("s[\"db2\"] = " + str(s["db2"]))