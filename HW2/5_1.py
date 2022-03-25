'''
Author: Xiang Pan
Date: 2022-02-23 23:08:49
LastEditTime: 2022-03-05 21:52:32
LastEditors: Xiang Pan
Description: 
FilePath: /HW2/5_1.py
@email: xiangpan@nyu.edu
'''
from cv2 import transform
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = train_transform

train_dataset = torchvision.datasets.FashionMNIST(root='./cached_datasets/FashionMNIST', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./cached_datasets/FashionMNIST', train=False, download=True, transform=test_transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

model = torch.hub.load('pytorch/vision:v0.10.0', model='googlenet', pretrained=False)

def train_model(model, train_dataloader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.cuda(), target.cuda()
        output = model
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print("Train Epoch: {} \tLoss: {:.6f}".format(batch_idx, loss.item()))

def test_model(model, test_dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_dataloader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss = test_loss / len(test_dataloader)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_dataloader.dataset), 100. * correct / len(test_dataloader.dataset)))


if __name__ == "__main__":
    train_model(model, train_dataloader)
    test_model(model, test_dataloader)