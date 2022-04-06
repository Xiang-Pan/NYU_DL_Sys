import argparse
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb
from pytorch_lightning.loggers import WandbLogger
import pandas as pd 
import torchmetrics

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_dropout=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.droup2 = nn.Dropout(0.5)

        self.use_dropout = use_dropout
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.fc1.in_features)
        if self.use_dropout:
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x).squeeze()
            x = self.droup2(x)
        else:
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x).squeeze()
        return x

class MLPLightningModule(pl.LightningModule):
    def __init__(self, hidden_size, output_size, optimizer_name='adam', use_dropout=False):
        super(MLPLightningModule, self).__init__()
        self.model = MLP(input_size=3*32*32, hidden_size=hidden_size, output_size=output_size, use_dropout=use_dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.Accuracy()
        self.optimizer_name = optimizer_name
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        # print(y_hat.shape)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def train_dataloader(self):
        train_dataset = datasets.CIFAR10(root='./cached_datasets/CIFAR10', train=True, download=True, transform=transforms.ToTensor())
        return torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=20)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return {'val_loss': loss}
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.acc_metric(y_hat, y)
        self.log('test_loss', loss, on_step=None, on_epoch=True)
        self.log('test_acc', acc, on_step=None, on_epoch=True)
        return {'test_loss': loss, 'test_acc': acc}
    def configure_optimizers(self):
        # print(self.optimizer_name, self.optimizer_name == "RMSprop")
        if self.optimizer_name == 'Adagrad':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=0.001)
        if self.optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=0.001)
        if self.optimizer_name == 'RMSprop+Nesterov':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=0.001, momentum=0.9)
        if self.optimizer_name == 'Adadelta':
            optimizer = torch.optim.Adadelta(self.parameters(), lr=0.001)
        if self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


test_dataloader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./cached_datasets/CIFAR10', train=False, download=True, transform=transforms.ToTensor()), batch_size=128, shuffle=True, num_workers=20)
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("xiang-pan/NYU_DL_Sys-HW3_Problem1")

summary_list, config_list, name_list = [], [], []
run_dict = {}
for run in runs: 
    name_list.append(run.name)
    if "Dropout" not in run.name:
        continue
    name = run.name.split("_")[0]
    run_dict[name] = run.id


def get_lastest_file_from_dir(dir):
    # get lastest file
    import os
    import glob
    file_list = glob.glob(dir + "/*")
    file_list.sort(key=os.path.getmtime)
    return file_list[-1]

def test_model(model_name):
    dir = f"./outputs/NYU_DL_Sys-HW3_problem1/{model_id}/checkpoints/"
    file = get_lastest_file_from_dir(dir)
    print(file)
    model = MLPLightningModule.load_from_checkpoint(file, hidden_size=1000, output_size=10)
    trainer = pl.Trainer(gpus=[0], max_epochs=200)
    res = trainer.test(model, test_dataloader)
    return res

if __name__ == '__main__':
    print(run_dict)
    res_dict = {}
    for op in ["Adagrad", "RMSprop", "RMSprop+Nesterov", "Adadelta", "Adam"]:
        print(op+"+Dropout")
        model_id = run_dict[op+"+Dropout"]
        res = test_model(model_id)
        res_dict[op+"+Dropout"] = res
        print(op, res)
    import json
    f = open("1_4.json", "w")
    json.dump(res_dict, f)