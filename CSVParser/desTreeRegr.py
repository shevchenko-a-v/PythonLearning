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
from DataPreprocessing import preprocess
from config import *

train_data = pd.read_csv(TRAIN_FILE, delimiter=",", encoding="Windows-1251")
test_data = pd.read_csv(TEST_FILE, delimiter=",", encoding="Windows-1251")
train_X = None
train_Y = None
test_X = None

(train_X,train_Y, test_X) = preprocess(train_data, test_data)


treeRegr = DecisionTreeRegressor(max_depth=20)
treeRegr.fit(train_X, train_Y)
pred_train_Y = treeRegr.predict(train_X)

# The mean squared error
mae = 0
for j in range(train_Y.shape[0]):
    if train_Y[j] != 0:
        mae += abs((train_Y[j] - pred_train_Y[j]) / pred_train_Y[j])

print("Mean squared error: %.5f" % (mae/train_Y.shape[0]))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(train_Y, pred_train_Y))


pred_test_Y = treeRegr.predict(test_X)

file = 'predict_results.csv'

myFile = open(file,'w')
i = 0
for j in range(pred_test_Y.shape[0]):
   myFile.write('%d,%f\n'%(i, pred_test_Y[i]))
   i += 1

myFile.close()
