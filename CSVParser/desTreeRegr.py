#!/usr/bin/python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
#from CSVParser import csv_parser
from CSVParser_textcolumns import csv_parser

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


def get_month_col(len):
    monthCol = np.empty((len,1), int)
    for i in range(monthCol.shape[0]):
        monthCol[i][0]=i%6
    return monthCol

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
    # add month column and clone other data values
    retVal=np.empty(shape=(result.shape[0]*6,result.shape[1]), dtype=np.float64)
    for ri,row in enumerate(result):
        for i in range(6):
            for ci,x in enumerate(row):
                retVal[ri*6+i][ci] = row[ci]
                
    retVal = np.c_[retVal, get_month_col(retVal.shape[0])]
    return retVal
        
train_data = pd.read_csv("C:\\NeuralNet\\OIL\\task1_data\\train_1.8.csv", delimiter=",", encoding="Windows-1251")
test_data = pd.read_csv("C:\\NeuralNet\\OIL\\task1_data\\test_1.9.csv", delimiter=",", encoding="Windows-1251")
train_X = None
train_Y = None
test_X = None
test_Y = None
trainArgs = None
testArgs = None

(train_X, train_Y, trainArgs) = csv_parser(train_data, False)
(test_X, test_Y, testArgs) = csv_parser(test_data, True)

train_X = np.c_[train_X, get_month_col(train_X.shape[0])]

test_X = update_test_data_structure(train_X, test_X, trainArgs, testArgs)

treeRegr = DecisionTreeRegressor(max_depth=20)
treeRegr.fit(train_X, train_Y)
pred_train_Y = treeRegr.predict(train_X)


#myFile1 = open("trainY4.csv",'w')
#myFile2 = open("predY4.csv",'w')

# The mean squared error
mae = 0
for j in range(train_Y.shape[0]):
    if train_Y[j] != 0:
        mae += abs((train_Y[j] - pred_train_Y[j]) / pred_train_Y[j])

print("Mean squared error: %.5f" % (mae/train_Y.shape[0]))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(train_Y, pred_train_Y))


pred_test_Y = treeRegr.predict(test_X)

#file = 'predict_results.csv'

#myFile = open(file,'w')
#i = 0
#for j in range(pred_test_Y.shape[0]):
#   myFile.write('%d,%f\n'%(i, pred_test_Y[i]))
#   i += 1

#myFile.close()
