from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from CSVParser import csv_parser
from math import fabs

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.fc1 = nn.Linear(88, 1)
        #self.conv1 = nn.Conv1d(1, 10, kernel_size=5)
        #self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        #self.fc1 = nn.Linear(10*44,20)
        #self.fc2 = nn.Linear(20, 1)

        self.fc1 = nn.Linear(88, 1)
        #self.fc2 = nn.Linear(40, 10)
        #self.fc3 = nn.Linear(10, 1)
        
        #self.fc1 = nn.Linear(88, 20)
        #self.fc2 = nn.Linear(20, 1)

    def forward(self, x):   
        return self.fc1(x)
        
        #x = F.relu(self.conv1(x))
        #x = self.pool(x)
        #x = x.view(-1,10*44)
        #x = F.relu(self.fc1(x))
        #return self.fc2(x)
        
        #out1 = F.relu(self.fc1(x))
        #out2 = F.relu(self.fc2(out1))
        #out3 = self.fc3(out2)        
        return out3

def train_old(model, device, X, Y, optimizer):
    model.train()
    for batch_idx, X_row in enumerate(X):
        data = X_row.to(device)
        target = Y[batch_idx].to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.l1_loss(output, target)
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Iteration:{} Loss: {:.6f}'.format(
            batch_idx , loss.item()))
def train_loader(model, decvice, loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.l1_loss(output.squeeze(1), target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            0, batch_idx * len(data), len(loader.dataset),
            100. * batch_idx / len(loader), loss.item()))

def train(model, device, X, Y, optimizer):
    for i in range(int(X.size()[0]/6)):
        data = X.narrow(0,i*6, 6)
        target = Y.narrow(0, i*6, 6)
        model.train()
        for j in range(data.size()[0]):
            data_x,target_y = data[j], target[j]
            optimizer.zero_grad()
            output = model(data_x)
            loss = F.l1_loss(output, target_y)
            #loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            print('Train Iteration:{} Loss: {:.6f}'.format(
                i , loss.item()))

def test_loader(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data[0].to(device)
            output = model(data)
            print(output[0].item())
            
def test(model, device, X):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X_row in X:
            data = X_row.to(device)
            output = model(data)
            print(output[0])


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

train_X_torch = torch.from_numpy(train_X)
train_Y_torch = torch.from_numpy(train_Y)
test_X_torch = torch.from_numpy(test_X)

loader = torch.utils.data.TensorDataset(train_X_torch, train_Y_torch)
train_loader_data = torch.utils.data.DataLoader(loader, batch_size=4)
loader2 = torch.utils.data.TensorDataset(test_X_torch)
test_loader_data = torch.utils.data.DataLoader(loader2, batch_size=20)

device = torch.device("cpu")
model = Net().to(device).double()
#optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.1)

for i in range(20):
    #train(model, device, train_X_torch, train_Y_torch, optimizer)
    train_loader(model, device, train_loader_data, optimizer)
    print('epoch {}'.format(i))

test_loader(model, device, test_loader_data)
