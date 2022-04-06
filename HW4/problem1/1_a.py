'''
Created by: Xiang Pan
Date: 2022-04-06 19:01:24
LastEditors: Xiang Pan
LastEditTime: 2022-04-06 19:50:52
Email: xiangpan@nyu.edu
FilePath: /HW4/problem1/1_a.py
Description: 
'''
import argparse
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100
from torchvision.models import resnet50
import torch
import torch.nn as nn
import wandb
import pytorch_lightning as pl
import torchmetrics
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning.loggers import WandbLogger

class ResNetLightningModule(pl.LightningModule):
    def __init__(self):
        super(ResNetLightningModule, self).__init__()
        self.model = resnet50(pretrained=True, progress=True)
        self.model.fc = nn.Linear(2048, 100)

        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.Accuracy()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.acc_metric(y_hat, y)
        self.log('train_loss', loss, prog_bar=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=False, on_epoch=True)
        logs = {'train_loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.acc_metric(y_hat, y)
        self.log('val_loss', loss, prog_bar=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=False, on_epoch=True)

    def configure_optimizers(self):
        lr = 0.001
        momentum = 0.9
        batch_size = 64
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # multi_step_lr_scheduler
        # For example, if training for 200 epochs, first drop can happen at epoch 60, second at epoch 120 and third at epoch 180.
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 180], gamma=0.1)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    # prepare_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = CIFAR100(root='./cached_datasets', train=True, transform=train_transform, download=True)
    test_dataset = CIFAR100(root='./cached_datasets', train=False, transform=test_transform, download=True)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    run = wandb.init(group='1_a')
    log_name = '1_a_lr' + args.lr
    logger = WandbLogger(project='HW4_1', save_dir="./outputs", name=log_name)

    trainer = pl.Trainer(gpus=args.gpus, max_epochs=200, logger=logger)
    model = ResNetLightningModule()
    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)
