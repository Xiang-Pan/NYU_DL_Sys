import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim

torch.manual_seed(0)

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

train_dataset = torchvision.datasets.CIFAR10(root='./cached_datasets',train=True, download=True, transform = train_transforms)
test_dataset = torchvision.datasets.CIFAR10(root='./cached_datasets', train=False, download=True, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=2)

total_step = len(trainloader)
classes = test_dataset.classes

for run_no in range(1,6): 
    
    print("run_no: ", run_no)
    
    model = torchvision.models.resnet50(pretrained=True,progress=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
               
    f = open(str(run_no)+"_"+GPU+".csv", "w")
    f.write('epoch,acc test,acc train,time\n')
    
    start = time.time()
    for epoch in range(num_epochs):
        correct_train = 0
        total_train = 0
        
        correct = 0
        total =0
        acc = 0
        
        for i, (images, labels) in enumerate(trainloader, 0):
            model.train()
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            predicted=torch.argmax(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct_train = correct
            total_train = total
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            if (i) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss:{:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))

            # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            acc = 0
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                predicted=torch.argmax(outputs,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
            print('Accuracy of the model on the test images: {} %'.format(acc))
            f.write('%d,%.7f,%.7f,%.7f\n' % (epoch + 1, acc, 100*(correct_train*1.0/total_train), time.time() - start))
            
            model.train()
            if acc >=85:
                torch.save(model, "./problem5/"+GPU+str(run_no)+".h5")
                break
                
        
        
    end = time.time()
    print(end - start, "seconds")
    f.close()
    print('Finished Training '+ str(run_no))