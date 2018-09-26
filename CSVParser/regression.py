#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the variance score are also
calculated.

"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import numpy as np
import pandas as pd
from math import sqrt
from CSVParser import csv_parser

def getDifferences(a,b):
    diffKyes = set(a.keys()) ^ set(b.keys())
    only_a=list()
    only_b=list()
    for key in diffKyes:
        if key in trainArgs:
            only_a.append(a[key])
            #only_a[key]=a[key]
        elif key in testArgs:
            only_b.append(b[key])
            #only_b[key]=b[key]
    only_a.sort()
    only_b.sort()
    return (only_a, only_b)


def update_test_data_structure(train_data, test_data, trainArgs,testArgs):
    (onlyTrainArgs,onlyTestArgs) = getDifferences(trainArgs,testArgs)
    print(onlyTrainArgs)
    result=test_data
    removedCnt=0
    for idx in onlyTestArgs:
        result = np.delete(result,idx-removedCnt, 1)
        removedCnt+=1
    for idx in onlyTrainArgs:
        result = np.insert(result,idx,np.zeros(result.shape[0]),1)
    return result


train_data = pd.read_csv("C:\\NeuralNet\\OIL\\task1_data\\train_1.8.csv", delimiter=",", encoding="Windows-1251")
test_data = pd.read_csv("C:\\NeuralNet\\OIL\\task1_data\\test_1.9.csv", delimiter=",", encoding="Windows-1251")
train_X = None
train_Y = None
test_X = None
test_Y = None
trainArgs = None
testArgs = None

(train_X, train_Y, trainArgs) = csv_parser(train_data)
(test_X, test_Y, testArgs) = csv_parser(test_data)

test_X = update_test_data_structure(train_X, test_X, trainArgs, testArgs)

np.set_printoptions(threshold=np.nan)
#print(train_X)

# Create linear regression object
regr = linear_model.PassiveAggressiveRegressor()

# Train the model using the training sets
regr.fit(train_X, train_Y)

# Make predictions using the testing set
pred_train_Y = regr.predict(train_X)

# The coefficients
# print('Coefficients: \n', regr.coef_)
# print("X ", train_X)
# print("Y ", train_Y)
# print("pred_train_Y ", pred_train_Y)

# print("X dim", train_X.shape)
# print("Y dim", train_Y.shape)
# print("X dim 0", train_X.shape[0])
# print("Coeff dim 0", regr.coef_.shape[0])


# The mean squared error
mae = 0
for j in range(train_Y.shape[0]):
    if train_Y[j] != 0:
        mae += abs((train_Y[j] - pred_train_Y[j]) / train_Y[j])
        #mae += (train_Y[j] - pred_train_Y[j])*(train_Y[j] - pred_train_Y[j])
        #j += 1


print("Mean squared error: %.5f" % (mae/train_Y.shape[0]))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(train_Y, pred_train_Y))


print(train_X.shape, test_X.shape)
pred_test_Y = regr.predict(test_X)
file = 'predict_results.csv'

myFile = open(file,'w')
i = 0
for j in range(pred_test_Y.shape[0]):
   myFile.write('%d;%f\n'%(i, pred_test_Y[i]))
   i += 1

myFile.close()
