import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb
from pytorch_lightning.loggers import WandbLogger
import argparse
import sys
sys.path.append('..')
sys.path.append('.')
from models.resnet import ResNet18
import time

class ResNetLightningModule(pl.LightningModule):
    def __init__(self, batch_size=32 ,optimizer_name='SGD'):
        super(ResNetLightningModule, self).__init__()
        self.model = ResNet18()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs}
    
    def on_train_epoch_end(self) -> None:
        return super().on_train_epoch_end()

    def train_dataloader(self):
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),#random crop
                                      transforms.RandomHorizontalFlip(0.5),#random flip
                                      transforms.ToTensor(), #convert to tensor
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#normalize
                                      ])
        train_dataset = datasets.CIFAR10(root='./cached_datasets/CIFAR10', train=True, download=True, transform=train_transform)

        return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    def test_dataloader(self):
        test_transform = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
            transforms.ToTensor(),])
        test_dataset = datasets.CIFAR10(root='./cached_datasets/CIFAR10', train=True, download=True, transform=test_transform)

        return torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        return {'val_loss': loss}
        
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}

        return {'val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        if self.optimizer_name == 'SGD':
            # Use SGD optimizers with 0.1 as the learning rate, momentum 0.9, weight decay 5e-4.
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
        return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=[1,2])
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--optimizer_name', type=str, default="SGD")

    args = parser.parse_args()
    run = wandb.init(group='2.1')
    optimizer_name = args.optimizer_name

    log_name = optimizer_name
    log_name += '_' + "dp"
    log_name += '_' + f"batch_size_{args.batch_size}"
    log_name += '_' + f"{len(args.gpus)}gpus"

    wandb.run.name = log_name
    logger = WandbLogger(project='HW3', save_dir="./outputs", name=log_name)
    trainer = pl.Trainer(gpus=args.gpus, 
                            max_epochs=2, 
                            logger=logger, 
                            strategy='dp', 
                            profiler="simple",
                            log_every_n_steps=1,)
    model = ResNetLightningModule(batch_size=args.batch_size)
    trainer.fit(model)