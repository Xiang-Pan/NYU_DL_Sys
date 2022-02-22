'''
Author: Xiang Pan
Date: 2022-02-22 00:54:59
LastEditTime: 2022-02-22 00:56:42
LastEditors: Xiang Pan
Description: 
FilePath: /NYU_DL_Sys/3_2.py
@email: xiangpan@nyu.edu
'''
from torch import nn


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