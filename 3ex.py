import numpy as np
import h5py
import matplotlib.pyplot as plt

# 加载我们自定义的工具函数
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
# 该函数用于初始化所有层的参数w和b
layers_dims = [12288, 20, 7, 5, 1]

def initialize_parameters_deep(layer_dims):
    """
    参数:
    layer_dims -- 这个list列表里面，包含了每层的神经元个数。
    例如，layer_dims=[5,4,3]，表示第一层有5个神经元，第二层有4个，最后一层有3个神经元
    
    返回值:
    parameters -- 这个字典里面包含了每层对应的已经初始化了的W和b。
    例如，parameters['W1']装载了第一层的w，parameters['b1']装载了第一层的b
    """
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims) # 获取神经网络总共有几层

    # 遍历每一层，为每一层的W和b进行初始化
    for l in range(1, L):
        # 构建并随机初始化该层的W。由我前面的文章《1.4.3 核对矩阵的维度》可知，Wl的维度是(n[l] , n[l-1]) 
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l-1])
        # 构建并初始化b
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        # 核对一下W和b的维度是我们预期的维度
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    #就是利用上面的循环，我们就可以为任意层数的神经网络进行参数初始化，只要我们提供每一层的神经元个数就可以了。       
    return parameters

def linear_forward(A, W, b):   
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b) # 将这些变量保存起来，因为后面进行反向传播时会用到它们
    
    return Z, cache #cache=(X,W,b)

def linear_activation_forward(A_prev, W, b, activation):
    """
    Arguments:
    A_prev -- 上一层得到的A，输入到本层来计算Z和本层的A。第一层时A_prev就是特征输入X
    W -- 本层相关的W
    b -- 本层相关的b
    activation -- 两个字符串，"sigmoid"或"relu"，指示该层应该使用哪种激活函数
    """
    
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "sigmoid": # 如果该层使用sigmoid        
        A = sigmoid(Z) 
    elif activation == "relu":
        A = relu(Z)
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, Z) # 缓存一些变量，后面的反向传播会用到它们((X,W,b),Z)

    return A, cache

def L_model_forward(X, parameters):
    """
    参数:
    X -- 输入的特征数据
    parameters -- 这个list列表里面包含了每一层的参数w和b
    """

    caches = []
    A = X
    
    # 获取参数列表的长度，这个长度的一半就是神经网络的层数。
    # 为什么是一半呢？因为列表是这样的[w1,b1,w2,b2...wl,bl],里面的w1和b1代表了一层
    L = len(parameters) // 2  
    
    # 循环L-1次，即进行L-1步前向传播，每一步使用的激活函数都是relu
    for l in range(1, L):
        print(l)
        A_prev = A
        print(A_prev)
        A, cache = linear_activation_forward(A_prev,
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)],
                                             activation='relu')
        caches.append(cache)# 把一些变量数据保存起来，以便后面的反向传播使用
        
    
    # 进行最后一层的前向传播，这一层的激活函数是sigmoid。得出的AL就是y'预测值
    AL, cache = linear_activation_forward(A, 
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          activation='sigmoid')
    caches.append(cache)
   
    assert(AL.shape == (1, X.shape[1]))
            
    return AL, caches
parameters=initialize_parameters_deep(layers_dims)
A,caches=L_model_forward(train_x,parameters)
print(A)
exit()
def compute_cost(AL, Y):
       
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    
    cost = np.squeeze(cost)# 确保cost是一个数值而不是一个数组的形式
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    """
    参数:
    dZ -- 后面一层的dZ
    cache -- 前向传播时我们保存下来的关于本层的一些变量(X,W,b)
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, cache[0].T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(cache[1].T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    参数:
    dA -- 本层的dA 
    cache -- 前向传播时保存的本层的相关变量
    activation -- 指示该层使用的是什么激活函数: "sigmoid" 或 "relu"
    """
    linear_cache, activation_cache = cache#(A,W,b) and Z
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    # 这里我们又顺带根据本层的dZ算出本层的dW和db以及前一层的dA
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    参数:
    AL -- 最后一层的A，也就是y'，预测出的标签
    Y -- 真实标签
    caches -- 前向传播时保存的每一层的相关变量，用于辅助计算反向传播
    """
    grads = {}
    L = len(caches) # 获取神经网络层数。caches列表的长度就等于神经网络的层数
    Y = Y.reshape(AL.shape) # 让真实标签的维度和预测标签的维度一致
    
    # 计算出最后一层的dA，前面文章我们以及解释过，最后一层的dA与前面各层的dA的计算公式不同，
    # 因为最后一个A是直接作为参数传递到成本函数的，所以不需要链式法则而直接就可以求dA（A相当于成本函数的偏导数）
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # 计算最后一层的dW和db，因为最后一层使用的激活函数是sigmoid
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(
                                                                                            dAL, 
                                                                                            current_cache,
                                                                                            activation = "sigmoid")

    # 计算前面L-1层到第一层的每层的梯度，这些层都使用relu激活函数
    for c in reversed(range(1,L)): # reversed(range(1,L))的结果是L-1,L-2...1。是不包括L的。第0层是输入层，不必计算。
        # 这里的c表示当前层
        grads["dA" + str(c-1)], grads["dW" + str(c)], grads["db" + str(c)] = linear_activation_backward(
            grads["dA" + str(c)], 
            caches[c-1],
            # 这里我们也是需要当前层的caches，但是为什么是c-1呢？因为grads是字典，我们从1开始计数，而caches是列表，
            # 是从0开始计数。所以c-1就代表了c层的caches。数组的索引很容易引起莫名其妙的问题，大家编程时一定要留意。
            activation = "relu")

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Arguments:
    parameters -- 每一层的参数w和b 
    grads -- 每一层的梯度
    learning_rate -- 是学习率，学习步进
    """
    
    L = len(parameters) // 2 # 获取层数。//除法可以得到整数

    for l in range(1,L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        
    return parameters


# print(train_x)
# exit()
def dnn_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False): 
    """    
    参数:
    X -- 数据集
    Y -- 数据集标签
    layers_dims -- 指示该深度神经网络用多少层，每层有多少个神经元
    learning_rate -- 学习率
    num_iterations -- 指示需要训练多少次
    print_cost -- 指示是否需要在将训练过程中的成本信息打印出来，好知道训练的进度好坏。
    
    返回值:
    parameters -- 返回训练好的参数。以后就可以用这些参数来识别新的陌生的图片
    """

    np.random.seed(1)
    costs = []                  

    # 初始化每层的参数w和b
    parameters = initialize_parameters_deep(layers_dims)
    
    # 按照指示的次数来训练深度神经网络
    for i in range(0, num_iterations):
        # 进行前向传播
        AL, caches = L_model_forward(X, parameters)
        # 计算成本
        cost = compute_cost(AL, Y)
        # 进行反向传播
        grads = L_model_backward(AL, Y, caches)
        # 更新参数，好用这些参数进行下一轮的前向传播
        parameters = update_parameters(parameters, grads, learning_rate)

        # 打印出成本
        if i==0:
            print(AL.shape)
        if i % 500 == 0:
            if print_cost and i > 0:
                print ("训练%i次后成本是: %f" % (i, cost))
            costs.append(cost)
            
    # 画出成本曲线图
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

# 根据上面的层次信息来构建一个深度神经网络，并且用之前加载的数据集来训练这个神经网络，得出训练后的参数
parameters = dnn_model(train_x, train_y, layers_dims, num_iterations=2000, print_cost=True)