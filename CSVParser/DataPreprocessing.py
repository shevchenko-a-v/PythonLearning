import numpy as np
#from CSVParser import csv_parser
from CSVParser_textcolumns import csv_parser

def get_month_col(len):
    monthCol = np.empty((len,1), int)
    for i in range(monthCol.shape[0]):
        monthCol[i][0]=i%6
    return monthCol

def getDifferences(a,b):
    diffKyes = set(a.keys()) ^ set(b.keys())
    only_a=list()
    only_b=list()
    for key in diffKyes:
        if key in trainArgs:
            only_a.append(a[key])
        elif key in testArgs:
            only_b.append(b[key])
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
    # add month column and clone other data values
    retVal=np.empty(shape=(result.shape[0]*6,result.shape[1]), dtype=np.float64)
    for ri,row in enumerate(result):
        for i in range(6):
            for ci,x in enumerate(row):
                retVal[ri*6+i][ci] = row[ci]
                
    retVal = np.c_[retVal, get_month_col(retVal.shape[0])]
    return retVal
        

def preprocess(train_data, test_data):
    # train data
    train_X = None
    train_Y = None
    trainArgs = None
    (train_X, train_Y, trainArgs) = csv_parser(train_data, False)
    train_X = np.c_[train_X, get_month_col(train_X.shape[0])]
    #test data
    test_X = None
    test_Y = None 
    testArgs = None 
    (test_X, test_Y, testArgs) = csv_parser(test_data, True)
    test_X = update_test_data_structure(train_X, test_X, trainArgs, testArgs)
    return train_X, train_Y, test_X