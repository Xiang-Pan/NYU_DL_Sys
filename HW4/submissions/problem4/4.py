import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim


networks = ['resnet18', 'resnet20', 'resnet32','resnet44','resnet56','resnet50']

GPU = torch.cuda.get_device_name(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device,"\n\n")

BATCH_SIZE = 128

train_transforms = transforms.Compose(
    [transforms.ToTensor(),transforms.RandomHorizontalFlip(),torchvision.
     transforms.RandomRotation((-15, +15)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True, download=True, transform = train_transforms)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=2)

classes = test_dataset.classes
# print(classes)


import torch
import torch.nn as nn
import torch.nn.functional as F


class Block_Plain(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block_Plain, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
            # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet_Plain(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet_Plain, self).__init__()
        # From table 1 in ResNet paper
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
            
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        
        for i in range(num_residual_blocks - 1):
            
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        
        return nn.Sequential(*layers)


class block(nn.Module):
    def __init__(self, filters, subsample=False):
        super().__init__()
        
        s = 0.5 if subsample else 1.0
        
        self.conv1 = nn.Conv2d(int(filters*s), filters, kernel_size=3, 
                               stride=int(1/s), padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters, track_running_stats=True)
        self.relu2 = nn.ReLU()

        # Shortcut downsampling
        self.downsample = nn.AvgPool2d(kernel_size=1, stride=2)
        
    def shortcut(self, z, x):

        if x.shape != z.shape:
            d = self.downsample(x)
            p = torch.mul(d, 0)
            return z + torch.cat((d, p), dim=1)
        else:
            return z + x        
    
    def forward(self, x, shortcuts=False):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu1(z)
        
        z = self.conv2(z)
        z = self.bn2(z)
        
        if shortcuts:
            z = self.shortcut(z, x)

        z = self.relu2(z)
        
        return z
    


class ResNet(nn.Module):
    def __init__(self, n, shortcuts=True):
        super().__init__()
        self.shortcuts = shortcuts
        
        # Input
        self.convIn = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnIn   = nn.BatchNorm2d(16, track_running_stats=True)
        self.relu   = nn.ReLU()
        
        # Stack1
        self.stack1 = nn.ModuleList([block(16, subsample=False) for _ in range(n)])

        # Stack2
        self.stack2a = block(32, subsample=True)
        self.stack2b = nn.ModuleList([block(32, subsample=False) for _ in range(n-1)])

        # Stack3
        self.stack3a = block(64, subsample=True)
        self.stack3b = nn.ModuleList([block(64, subsample=False) for _ in range(n-1)])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcOut   = nn.Linear(64, 10, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)
                
        
    def forward(self, x):     
        z = self.convIn(x)
        z = self.bnIn(z)
        z = self.relu(z)
        
        for l in self.stack1: z = l(z, shortcuts=self.shortcuts)
        
        z = self.stack2a(z, shortcuts=self.shortcuts)
        for l in self.stack2b: 
            z = l(z, shortcuts=self.shortcuts)
        
        z = self.stack3a(z, shortcuts=self.shortcuts)
        for l in self.stack3b: 
            z = l(z, shortcuts=self.shortcuts)

        z = self.avgpool(z)
        z = z.view(z.size(0), -1)
        z = self.fcOut(z)
        return self.softmax(z)



def eval_model(val_loader):
    acc = 0
    running_loss = 0.0
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        acc += (torch.sum(labels == outputs.argmax(dim=1))/BATCH_SIZE)
        vf.write('%d,%5d,%.7f,%.7f\n' % (epoch + 1, i + 1, loss.item(),acc/(i+1)))
        running_loss += loss.item()
    print('Epoch: %d, Step: %5d Val loss: %.9f, Acc: %.9f' %
        (epoch + 1, i + 1, running_loss / 50,acc/(i+1)))
    running_loss = 0.0


img_channels = 3
num_classes = 10

def create_model(n_layers, num_classes=10):
    if n_layers==18:
        net = ResNet_Plain(18, Block_Plain, img_channels, num_classes)
    elif n_layers ==20:
        net = ResNet(3, shortcuts=True)
    elif n_layers ==32:
        net = ResNet(5, shortcuts=True)
    elif n_layers ==44:
        net = ResNet(7, shortcuts=True)
    elif n_layers ==56:
        net = ResNet(9, shortcuts=True)
    # for question 4.2    
    elif n_layers==50:
        net = ResNet_Plain(50, Block_Plain, img_channels, num_classes)
    return net



EPOCH = 350

lr = 0.1 # authors cite 0.1
momentum = 0.9
weight_decay = 0.0001 
milestones = [82, 123]
gamma = 0.1


# criterion = torch.nn.NLLLoss()

criterion = nn.CrossEntropyLoss()

layers = [18,20,32,44,56,50]
networks = ["resnet"+str(i) for i in layers]

for NETWORK_ID in range(len(layers)):
    
    print(networks[NETWORK_ID])
        
    net = create_model(layers[NETWORK_ID]).to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
#     optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
    f = open(networks[NETWORK_ID]+"_"+GPU+".csv", "w")
    f.write('epoch, step, loss, acc\n')

    vf = open(networks[NETWORK_ID]+"_"+GPU+"_val.csv", "w")
    vf.write('epoch, step, loss, acc\n')

    for epoch in range(EPOCH):
        acc = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc += (torch.sum(labels == outputs.argmax(dim=1))/BATCH_SIZE)
            f.write('%d,%5d,%.7f,%.7f\n' % (epoch + 1, i + 1, loss.item(),acc/(i+1)))
            running_loss += loss.item()


            if i % 50 == 0: # print every 50 iterations
                print('Epoch: %d, Step: %5d loss: %.9f, Acc: %.9f' %
                (epoch + 1, i + 1, running_loss / 50,acc/(i+1)))
                running_loss = 0.0
                eval_model(testloader)

#         if epoch % 50 == 49:
#             torch.save(net.state_dict(),networks[NETWORK_ID]+"_"+GPU+"_"+str(epoch+1)+".pt")

    f.close()
    vf.close()
    print('Finished Training '+ networks[NETWORK_ID])