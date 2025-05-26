import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
data=datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
test=datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(data, batch_size=64, shuffle=True)
test_loader=DataLoader(test, batch_size=64, shuffle=True)
train_iterator = iter(train_loader)
class convNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1)
        self.L2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.L3=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.L4=nn.MaxPool2d(kernel_size=2,stride=2)
        self.L5=nn.Linear(400,120)
        self.L6=nn.Linear(120,100)
        self.L7=nn.Linear(100,84)
        self.outLayer=nn.Linear(84,10)
    def forward(self,img):
        out=F.leaky_relu(self.L1(img))
        out=self.L2(out)
        out=F.leaky_relu(self.L3(out))
        out=self.L4(out)
        out=out.view(-1,400)
        out=F.relu(self.L5(out))
        out=F.relu(self.L6(out))
        out=F.relu(self.L7(out))
        out=self.outLayer(out)
        return out
model=convNET()
l_type=nn.CrossEntropyLoss()
lr=0.01
optim=torch.optim.SGD(model.parameters(),lr,momentum=0.9)
num_epochs=20
for i in range(num_epochs):
    model.train()
    running_loss=0.0
    for images,labels in train_loader:
        outputs=model(images)
        loss=l_type(outputs,labels)
        loss.backward()
        optim.step()
        optim.zero_grad()
        running_loss+=loss.item()
    print(f"Epoch[{i+1}/{num_epochs}],Loss:{running_loss/len(train_loader):.4f}")    
model.eval()        
correct=0
total=0
with torch.no_grad():
    for images, labels in test_loader:
        outputs=model(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

print(f"Test Accuracy:{100*correct/total:.2f}%")      