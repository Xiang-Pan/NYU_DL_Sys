'''
Author: Xiang Pan
Date: 2022-03-27 12:53:32
LastEditTime: 2022-03-27 13:04:15
LastEditors: Xiang Pan
Description: 
FilePath: /HW3/problem4/4_2.py
@email: xiangpan@nyu.edu
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torch
from models.resnet44 import resnet44
import torchvision.datasets as datasets
import torchmetrics
import torchvision.transforms as transforms


class ResNetLightningModule(pl.LightningModule):
    def __init__(self, batch_size=64 ,optimizer_name='SGD', aug_method="simple_aug"):
        super(ResNetLightningModule, self).__init__()
        self.model = resnet44()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.aug_method = aug_method
        self.acc_metric = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self.model(x)
        loss = self.loss_fn(logit, y)
        train_acc = self.acc_metric(logit.argmax(dim=-1), y)
        self.log('train_acc', train_acc, prog_bar=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs}

    def train_dataloader(self):
        if self.aug_method == "simple_aug":
            train_transform = transforms.Compose([# 4 pixels are padded on each side, 
                                                transforms.Pad(4),
                                                # a 32×32 crop is randomly sampled from the 
                                                # padded image or its horizontal flip.
                                                transforms.RandomHorizontalFlip(0.5),
                                                transforms.RandomCrop(32),
                                                transforms.ToTensor()
                                            ])
        elif self.aug_method == "cutout":
            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),#random crop
                                        transforms.RandomHorizontalFlip(0.5),#random flip
                                        transforms.ToTensor(), #convert to tensor
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#normalize
                                        ])
        else:
            raise ValueError("Invalid augmentation method")
        train_dataset = datasets.CIFAR10(root="./cached_datasets/CIFAR10", train=True, transform=train_transform, download=True)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])
        self.train_dataset.transform = train_transform
        self.val_dataset.transform = transforms.Compose([transforms.ToTensor(),])
        
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    def validation_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def test_dataloader(self):
        test_transform = transforms.Compose([
            transforms.ToTensor(),])
        test_dataset = datasets.CIFAR10(root='./cached_datasets/CIFAR10', train=False, download=True, transform=test_transform)

        return torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        val_acc = self.acc_metric(y_hat.argmax(dim=-1), y)
        self.log('val_acc', val_acc, prog_bar=False, on_epoch=True)
        val_error = 1 - val_acc
        self.log('val_error', val_error, prog_bar=False, on_epoch=True)

        return {'val_loss': loss}
        

    def configure_optimizers(self):
        if self.optimizer_name == 'SGD':
            lr = 0.1 # authors cite 0.1
            momentum = 0.9
            weight_decay = 0.0001 
            # Use SGD optimizers with 0.1 as the learning rate, momentum 0.9, weight decay 0.0001
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        return optimizer

import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--batch_size', type=int, default='64')
    parser.add_argument('--optimizer_name', type=str, default="SGD")
    parser.add_argument('--aug_method', type=str, default="simple_aug")

    args = parser.parse_args()
    run = wandb.init(group='4.2')
    optimizer_name = args.optimizer_name

    log_name = args.aug_method

    wandb.run.name = log_name
    logger = WandbLogger(project='HW3', save_dir="./outputs", name=log_name)
    trainer = pl.Trainer(gpus=args.gpus, 
                            max_epochs=100, 
                            logger=logger, 
                            profiler="simple",)
    model = ResNetLightningModule(batch_size=args.batch_size, optimizer_name=optimizer_name, aug_method=args.aug_method)
    trainer.fit(model, [model.train_dataloader(), model.validation_dataloader()])