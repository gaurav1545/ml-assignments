# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:08:54 2019

@author: Skanda
"""

import numpy as np
import pandas as pd
import scipy

def sigmoid(x):
  return scipy.special.expit(x)

def logistic_regression(data):
    x = np.matrix(data.iloc[:, [5,0,1,2,3]])
    y = np.array(data.iloc[:, 4])
    y = np.expand_dims(y, axis = 1)
    w = np.matrix(np.random.rand(5))
    w = w.T
    alpha = 1e-2
    for k in range(250):
        y_preds = sigmoid(np.matmul(x,w))
        #y_preds[y_preds >= 0.5] = 1
        #y_preds[y_preds != 1] = 0
        update = np.dot(x.T, (y_preds - y))
        update *= alpha
        w -= update
        if (k+1) % 50 == 0:
            alpha /= 10
    y_preds[y_preds >= 0.5] = 1
    y_preds[y_preds != 1] = 0
    return np.equal(y, y_preds).astype(int).sum() / len(y)


def regularized_logistic_regression(data, epochs, lamda):
    x = np.matrix(data.iloc[:, [5,0,1,2,3]])
    y = np.array(data.iloc[:, 4])
    y = np.expand_dims(y, axis = 1)
    w = np.matrix(np.random.rand(5))
    w = w.T
    alpha = 1e-2
    for k in range(epochs):
        y_preds = sigmoid(np.matmul(x,w))
        #y_preds[y_preds >= 0.5] = 1
        #y_preds[y_preds != 1] = 0
        update = np.dot(x.T, (y_preds - y)) + lamda * w
        update *= alpha
        w -= update
        if (k+1) % 50 == 0:
            alpha /= 10
    y_preds[y_preds >= 0.5] = 1
    y_preds[y_preds != 1] = 0
    return np.equal(y, y_preds).astype(int).sum() / len(y)



def main():
    data = pd.read_csv('data_logistic.txt', header=None)
    data = data.assign(bias = np.zeros(len(data)) + 1)
    acc = logistic_regression(data)
    print(acc)
    epochs = 100
    lamda = 0.5
    acc = regularized_logistic_regression(data, epochs, lamda)
    print(acc)


if __name__ == '__main__':
    main()
