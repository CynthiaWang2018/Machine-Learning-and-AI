# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:58:28 2019

@author: Cynthia
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
# 为随机分割训练集和测试集
import math
import warnings
import matplotlib.pyplot as plt
#%matplotlib inline
warnings.filterwarnings('ignore')
# 定义样本被划分为某类的概率
def prob(i,theta, x):
    '''
    i 为第i个样本
    theta 为模型参数
    x 为第i个样本i.e.  x.iloc[i]
    '''
    theta_i = np.matrix(theta[i])
    x = np.matrix(x)
    numerator = math.exp(np.dot(theta_i, x.T))
    denominator = 0
    for j in range(k): # k为种类数
        theta_j = np.matrix(theta[j])
        denominator += math.exp(np.dot(theta_j, x.T))
    return numerator / denominator
# 定义示性函数
def I(x, y):
    if x == y:return 1
    return 0
# 定义梯度
def grad(j, theta):
    '''
    j 取1，2，3
    theta 为模型参数
    '''
    grad = np.array([0 for i in range(n+1)]) # 加1是因为多了一列bias的系数
    #grad : array([0, 0, 0, 0, 0])
    for i in range(m): # m 为样本个数 150
        tmp = I(y[i], j) - prob(j, theta, x.loc[i])
        grad += x.loc[i] * tmp
    return -grad / m
# 更新参数
def para(theta, alpha, iteration):# alpha = 0.0001, iteration = 1000
    '''
    theta 为参数
    alpha 为learning rate
    iteration 为迭代次数
    '''
    c = 0
    costs = []
    for iter in range(iteration):
        for j in range(k):
            theta[j] = theta[j] - alpha * grad(j, theta)
        # 如果时间太长，则采用矩阵乘法
        for i in range(m):
            for j in range(k):
                c += I(y[i], j) * np.log(prob(j, theta, x.loc[i]))
        c = -c / m
        costs.append(c)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'b')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Cost vs. Iteration')
    print('running iterations')
    return theta
# 计算属于每一类的概率
def h(x):
    '''
    x 为样本第i行
    '''
    x = np.matrix(x)
    h_matrix = np.empty((k, 1))
    denominator = 0
    for j in range(k):
        denominator += math.exp(np.dot(theta[j], x.T))
    for i in range(k):
        h_matrix[i] = math.exp(np.dot(theta[i], x.T))
    return h_matrix / denominator
# 导入数据
iris = pd.read_csv('./iris.csv')
# 分割为训练集，测试集
train, test = train_test_split(iris, test_size = 0.3,random_state=1)
train = train.reset_index()
test = test.reset_index()

x = train[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = train['variety']
y = y.map({'Setosa':0,'Versicolor':1,'Virginica':2})
n = x.shape[1]
m = x.shape[0]
k = len(y.unique())
x['bias'] = np.ones(x.shape[0])
theta = np.empty((k, n+1))

alpha = 0.01
iteration = 1000
theta = para(theta, alpha, iteration)

x_test = test[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
n = x_test.shape[1]
m = x_test.shape[0]

y_test = test['variety']
k = len(y_test.unique())
y_test =y_test.map({'Setosa':0,'Versicolor':1,'Virginica':2})
y_test.value_counts()
x_test['bias'] = np.ones(x_test.shape[0])

for index,row in x_test.iterrows():
    h_index = h(row)
    prediction = int(np.where(h_index == h_index.max())[0]) # 因为最大值可能不止一个，所以此处返回第一个
    x_test.loc[index,'prediction'] = prediction
    
results = x_test
results['actual'] = y_test

diff = results['prediction'] == results['actual']
correct = diff.value_counts()[1]
acc = correct/len(results)
print(acc)