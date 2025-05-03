import os
from torch.utils.data import Dataset
import glob
import cv2
from PIL import Image
import torch

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU = torch.cuda.get_device_name(0)

BATCH_SIZE = 128
num_epochs = 350
criterion = nn.CrossEntropyLoss()
lr=0.001
momentum=0.9

train_transforms = transforms.Compose(
    [transforms.ToTensor(),transforms.RandomHorizontalFlip(),torchvision.
     transforms.RandomRotation((-15, +15)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class CustomDataset(Dataset):
    def __init__(self):
    
        self.imgs_path = "./problem5/labeled_images"
        file_list = os.walk(self.imgs_path)

        # print(file_list)
        self.data = []
        self.class_map = {"airplane": 0, "automobile": 1, "bird": 2, "cat" : 3,"deer": 4, "dog": 5, "frog": 6,"horse": 7, "ship": 8, "truck": 9}
        self.class_list = list(self.class_map.keys())


        for c in self.class_list:
            class_path = os.path.join(self.imgs_path, c)
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append((img_path, c))

        
        self.img_dim = (32, 32)
    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        class_id = self.class_map[class_name]
        img = test_transforms(img)
        class_id = torch.tensor(class_id)
        
        return img, class_id


wild_dataset = CustomDataset()
test_loader = torch.utils.data.DataLoader(dataset=wild_dataset,batch_size=BATCH_SIZE)

v100_acc = []
rtx_acc = []
GPUS = ["problem5/Tesla V100-PCIE-32GB"]

for GPU in GPUS:
    for run_no in range(1,6):
        filename = GPU+str(run_no)+".h5"
        print(f'Model: {filename}')
        model = torch.load(filename)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            acc = 0
            for images, labels in test_loader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                predicted=torch.argmax(outputs,1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
                if '100' in filename.lower():
                    v100_acc.append(acc)
                else:
                    rtx_acc.append(acc)

            print("Total Images:", total, "Correctly classified: ", correct)
            print('Accuracy of the model on the test images: {} %'.format(acc))
            print()