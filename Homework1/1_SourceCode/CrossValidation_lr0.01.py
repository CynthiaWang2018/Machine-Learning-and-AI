# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:44:12 2019

@author: Cynthia
"""


import numpy as np
import pandas as pd
import math
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

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

def I(x, y):
    if x == y:return 1
    return 0

def grad(j, theta):
    '''
    j 取1，2，3
    theta 为模型参数
    '''
    grad = np.array([0 for i in range(n)]) 
    #grad : array([0, 0, 0, 0, 0])
    for i in range(m): # m 为样本个数 
        tmp = I(y[i], j) - prob(j, theta, x.loc[i])
        grad += x.loc[i] * tmp
    return -grad / m

def para(theta, alpha, iteration):# alpha = 0.01, iteration = 500
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

iris = pd.read_csv('./iris.csv')
from sklearn.cross_validation import train_test_split
#train, test = train_test_split(iris, test_size = 0.3)
train, test = train_test_split(iris, test_size = 0.3,random_state=1)
train = train.reset_index()
test = test.reset_index()

x = train[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = train['variety']
y = y.map({'Setosa':0,'Versicolor':1,'Virginica':2})

x['bias'] = np.ones(x.shape[0])

x1 = x[0:35]
x2 = x[35:70]
x3 = x[70:105]
y1 = y[0:35]
y2 = y[35:70]
y3 = y[70:105]

# Fold1------------------------------------------------------------------------
frames = [x1,x2]
x = pd.concat(frames)
x = x.reset_index()
x = x[['sepal.length', 'sepal.width', 'petal.length', 'petal.width','bias']]

frames2 = [y1,y2]
y = pd.concat(frames2)
y = y.reset_index()
y = y['variety']

n = x.shape[1]
m = x.shape[0]
k = len(y.unique())

theta = np.empty((k, n))

alpha = 0.01
# alpha = 0.1
# alpha = 0.001
# alpha = 0.0001
# 得出alpha为0.01时，准确率最高
iteration = 1000
theta = para(theta, alpha, iteration)

x_test = x3
x_test = x_test.reset_index()
x_test = x_test[['sepal.length', 'sepal.width', 'petal.length', 'petal.width','bias']]
n = x_test.shape[1]
m = x_test.shape[0]


y_test = y3
y_test = y_test.reset_index()
y_test = y_test['variety']
k = len(y_test.unique())

for index,row in x_test.iterrows():
    h_index = h(row)
    prediction = int(np.where(h_index == h_index.max())[0])
    x_test.loc[index,'prediction'] = prediction
    
results = x_test
results['actual'] = y_test

diff = results['prediction'] == results['actual']
correct = diff.value_counts()[1]
acc1 = correct/len(results)
print(acc1)
# Fold2------------------------------------------------------------------------
frames = [x1,x3]
x = pd.concat(frames)
x = x.reset_index()
x = x[['sepal.length', 'sepal.width', 'petal.length', 'petal.width','bias']]

frames2 = [y1,y3]
y = pd.concat(frames2)
y = y.reset_index()
y = y['variety']

n = x.shape[1]
m = x.shape[0]
k = len(y.unique())

theta = np.empty((k, n))

alpha = 0.01
iteration = 500
theta = para(theta, alpha, iteration)

x_test = x2
x_test = x_test.reset_index()
x_test = x_test[['sepal.length', 'sepal.width', 'petal.length', 'petal.width','bias']]
n = x_test.shape[1]
m = x_test.shape[0]

y_test = y2
y_test = y_test.reset_index()
y_test = y_test['variety']
k = len(y_test.unique())

for index,row in x_test.iterrows():
    h_index = h(row)
    prediction = int(np.where(h_index == h_index.max())[0])
    x_test.loc[index,'prediction'] = prediction
    
results = x_test
results['actual'] = y_test

diff = results['prediction'] == results['actual']
correct = diff.value_counts()[1]
acc2 = correct/len(results)
print(acc2)
# Fold3------------------------------------------------------------------------
frames = [x2,x3]
x = pd.concat(frames)
x = x.reset_index()
x = x[['sepal.length', 'sepal.width', 'petal.length', 'petal.width','bias']]

frames2 = [y2,y3]
y = pd.concat(frames2)
y = y.reset_index()
y = y['variety']

n = x.shape[1]
m = x.shape[0]
k = len(y.unique())

theta = np.empty((k, n))

alpha = 0.01
iteration = 500
theta = para(theta, alpha, iteration)

x_test = x1
x_test = x_test.reset_index()
x_test = x_test[['sepal.length', 'sepal.width', 'petal.length', 'petal.width','bias']]
n = x_test.shape[1]
m = x_test.shape[0]

y_test = y1
y_test = y_test.reset_index()
y_test = y_test['variety']
k = len(y_test.unique())


for index,row in x_test.iterrows():
    h_index = h(row)
    prediction = int(np.where(h_index == h_index.max())[0])
    x_test.loc[index,'prediction'] = prediction
    
results = x_test
results['actual'] = y_test

diff = results['prediction'] == results['actual']
correct = diff.value_counts()[1]
acc3 = correct/len(results)
print(acc3)