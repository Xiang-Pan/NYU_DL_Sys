'''
Author: Xiang Pan
Date: 2022-02-23 02:27:06
LastEditTime: 2022-03-07 03:39:00
LastEditors: Xiang Pan
Description: 
FilePath: /HW2/4_1.py
@email: xiangpan@nyu.edu
'''

import argparse
from email import parser
import numpy as np
import sys
import time

def f_function(x):
    x_1 = x[:,0]
    x_2 = x[:,1]
    return -(x_2+47)*np.sin(np.sqrt(np.abs(x_1/2+x_2+47))) - x_1*np.sin(np.sqrt(np.abs(x_1-x_2+47))) 

def y_function(x):
    n = len(x)
    return f_function(x) + np.random.normal(0, np.sqrt(0.3), size=(n))

def get_data(n):
    n = int(n)
    x_1 = np.random.uniform(low=-512, high=512, size=(n))
    x_2 = np.random.uniform(low=-512, high=512, size=(n))
    x = np.array([x_1,x_2]).T.astype(np.float32)
    print(x.shape)
    y = y_function(x).astype(np.float32)
    return x, y


import torch
import torch.nn as nn

def get_number_of_param(model):
    """get the number of param for every element"""
    count = 0
    for param in model.parameters():
        param_size = param.size()
        count_of_one_param = 1
        for dis in param_size:
            count_of_one_param *= dis
        print(param.size(), count_of_one_param)
        count += count_of_one_param
        print(count)
    print('total number of the model is %d'%count)


class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.x, self.y = get_data(n)
        print(self.x.shape, self.y.shape)
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)
        self.x = self.x / 512
        print(self.x, self.y.shape)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)):
            size = layer_sizes[i]
            self.layers.append(nn.Linear(size[0], size[1]))
            if i < len(layer_sizes)-1:
                self.layers.append(nn.BatchNorm1d(size[1]))
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RMSError(nn.Module):
    def __init__(self):
        super(RMSError, self).__init__()
    def forward(self, output, target):
        return torch.sqrt(torch.mean((output-target)**2))

def train_mlp(model):
    total = 100000
    train_size = total * 0.8
    test_size = total * 0.2
    train_dataset = MLPDataset(train_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True)
    test_dataset = MLPDataset(test_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    measure = RMSError()
    # loss_func = nn.RMSError()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    
    train_epoch = 2000
    for epoch in range(train_epoch):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = target.cuda()
            # print()
            optimizer.zero_grad()
            output = model(data).squeeze()
            # print(output, target)
            loss = measure(output, target)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        total_loss = total_loss/(batch_idx+1)
        print('Epoch: %d, Loss: %f' % (epoch, total_loss), end=' ')
        test_mlp(model, test_loader, measure)
        sys.stdout.flush()

def test_mlp(model, test_loader, measure):
    model.eval()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.cuda()
        target = target.cuda()
        output = model(data).squeeze()
        loss = measure(output, target)
        total_loss += loss.item()
    total_loss = total_loss/(batch_idx+1)
    print('Test: Loss: %f' % (total_loss))


def test_data():
    x, y = get_data(1)

    # assert f_function(x) == y
    print(x, y)

def main(number_layers, config_number):

    # two groups
    two_group_list = []
    
    layer_sizes = [[2,32],[32,1]] # 32
    two_group_list.append(layer_sizes)
    layer_sizes = [[2,64],[64,1]] # 64
    two_group_list.append(layer_sizes)
    layer_sizes = [[2,128],[128,1]] # 128
    two_group_list.append(layer_sizes)
    layer_sizes = [[2,256],[256,1]] # 256
    two_group_list.append(layer_sizes)
    layer_sizes = [[2,512],[512,1]] # 512
    two_group_list.append(layer_sizes)
    
    # three groups
    three_groups_list = []
    layer_sizes = [[2,32],[32,32],[32,1]] # 64
    three_groups_list.append(layer_sizes)
    layer_sizes = [[2,64],[64,64],[64,1]] # 128
    three_groups_list.append(layer_sizes)
    layer_sizes = [[2,128],[128,128],[128,1]] # 256
    three_groups_list.append(layer_sizes)
    layer_sizes = [[2,256],[256,256],[256,1]] # 512
    three_groups_list.append(layer_sizes)

    # four groups
    layer_sizes = [[2,32],[32,64],[64,32],[32,1]]       # 128
    four_groups_list = [layer_sizes]
    layer_sizes = [[2,64],[64,128],[128,64],[64,1]]     # 256
    four_groups_list.append(layer_sizes)
    layer_sizes = [[2,128],[128,256],[256,128],[128,1]] # 512
    four_groups_list.append(layer_sizes)


    if number_layers == 2:
        layer_sizes = two_group_list[config_number]
    elif number_layers == 3:
        layer_sizes = three_groups_list[config_number]
    elif number_layers == 4:
        layer_sizes = four_groups_list[config_number]

    # calculate the wall time
    f = open(f"./outputs/{number_layers}_layers_config_{config_number}.txt", "w")
    sys.stdout = f
    start_time = time.time()
    mlp = MLP(layer_sizes).cuda()
    print(get_number_of_param(mlp))
    # print(mlp)
    train_mlp(mlp)
    end_time = time.time()
    print('Wall time: %f' % (end_time - start_time))
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--number_layers', type=int, default=2)
    argparser.add_argument('--config_number', type=int, default=2)
    args = argparser.parse_args()
    main(args.number_layers, args.config_number)
