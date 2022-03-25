'''
Author: Xiang Pan
Date: 2022-02-22 00:54:59
LastEditTime: 2022-02-23 02:16:24
LastEditors: Xiang Pan
Description: 
FilePath: /HW2/3_2.py
@email: xiangpan@nyu.edu
'''
from torch import nn
import pandas as pd

class LeNet5_BN(nn.Module):
    def __init__(self):
        super(LeNet5_BN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(256, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.fc3_bn = nn.BatchNorm1d(10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y) # bn
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.conv2_bn(y) # bn
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.fc1_bn(y) # bn
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.fc2_bn(y) # bn
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.fc3_bn(y) # bn
        y = self.relu5(y)
        return y

class LeNet5_BN_Dropout(nn.Module):
    def __init__(self):
        super(LeNet5_BN_Dropout, self).__init__()
        self.input_dropout = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(256, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.relu3 = nn.ReLU()
        self.fc1_dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.relu4 = nn.ReLU()
        self.fc2_dropout = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(84, 10)
        self.fc3_bn = nn.BatchNorm1d(10)
        self.relu5 = nn.ReLU()
        self.fc3_dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y) # bn
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.conv2_bn(y) # bn
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)

        y = self.fc1(y)
        y = self.fc1_bn(y) # bn
        y = self.relu3(y)
        y = self.fc1_dropout(y)
        
        y = self.fc2(y)
        y = self.fc2_bn(y) # bn
        y = self.relu4(y)
        y = self.fc2_dropout(y)

        y = self.fc3(y)
        y = self.fc3_bn(y) # bn
        y = self.relu5(y)
        y = self.fc3_dropout(y)

        return y


class LeNet5_Dropout(nn.Module):
    def __init__(self):
        super(LeNet5_Dropout, self).__init__()
        self.input_dropout = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(256, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.relu3 = nn.ReLU()
        self.fc1_dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.relu4 = nn.ReLU()
        self.fc2_dropout = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(84, 10)
        self.fc3_bn = nn.BatchNorm1d(10)
        self.relu5 = nn.ReLU()
        self.fc3_dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        y = self.conv1(x)
        # y = self.conv1_bn(y) # bn
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        # y = self.conv2_bn(y) # bn
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)

        y = self.fc1(y)
        # y = self.fc1_bn(y) # bn
        y = self.relu3(y)
        y = self.fc1_dropout(y)
        
        y = self.fc2(y)
        # y = self.fc2_bn(y) # bn
        y = self.relu4(y)
        y = self.fc2_dropout(y)

        y = self.fc3(y)
        # y = self.fc3_bn(y) # bn
        y = self.relu5(y)
        y = self.fc3_dropout(y)
        return y


bn_layer_list = [model.conv1_bn, model.conv2_bn, model.fc1_bn, model.fc2_bn, model.fc3_bn]


df = pd.DataFrame(columns=['layer_name', 'mean', 'var'])

df["layer_name"] = ['conv1_bn', 'conv2_bn', 'fc1_bn', 'fc2_bn', 'fc3_bn']
df["mean"] = [layer.running_mean for layer in bn_layer_list]
df["var"] = [layer.running_var for layer in bn_layer_list]

bn_layer_dict = {
    "conv1_bn": model.conv1_bn,
    "conv2_bn": model.conv2_bn,
    "fc1_bn": model.fc1_bn,
    "fc2_bn": model.fc2_bn,
    "fc3_bn": model.fc3_bn
}


