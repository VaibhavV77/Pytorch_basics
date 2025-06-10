import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available - running on CPU")
data=datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
test=datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(data, batch_size=64, shuffle=True)
test_loader=DataLoader(test,batch_size=64,shuffle=False)
class ResidualConnect(nn.Module):
    def __init__(self,input_dim,outC,stride=1):
        super().__init__()
        self.L1=nn.Conv2d(in_channels=input_dim,out_channels=outC,kernel_size=3,stride=stride,padding=1)
        self.L2=nn.Conv2d(in_channels=outC,out_channels=outC,kernel_size=3,stride=1,padding=1)
        self.B2d=nn.BatchNorm2d(num_features=outC)
        self.B2d1=nn.BatchNorm2d(num_features=outC)
        if input_dim!=outC or stride!=1:
          self.downsample=nn.Sequential(nn.Conv2d(input_dim,outC,kernel_size=1,stride=stride),nn.BatchNorm2d(outC))
        else:
          self.downsample=None
    def forward(self,X):
        identity=X
        out=self.B2d(self.L1(X))
        out=F.relu(out)
        out=self.B2d1(self.L2(out))
        if self.downsample:
          identity = self.downsample(identity)
        out+=identity
        out=F.relu(out)
        return out
        
class RESnet18(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.LI=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.R1=nn.Sequential(ResidualConnect(64,64),ResidualConnect(64,64))
        self.R2=nn.Sequential(ResidualConnect(64,128),ResidualConnect(128,128))
        self.R3=nn.Sequential(ResidualConnect(128,256,stride=2),ResidualConnect(256,256,stride=2))
        self.R4=nn.Sequential(ResidualConnect(256,512,stride=2),ResidualConnect(512,512,stride=2))
        self.fc=nn.Linear(in_features=512,out_features=num_classes)
    def forward(self,img):
       out=self.LI(img)
       out=self.R1(out)
       out=self.R2(out) 
       out=self.R3(out)
       out=self.R4(out)
       out=F.adaptive_avg_pool2d(out,(1,1))
       out=torch.flatten(out,1)
       out=self.fc(out)
       return out
model=RESnet18(num_classes=10).to(device)
print(f"Model is on device: {next(model.parameters()).device}")
l_type=nn.CrossEntropyLoss()
lr=0.01
optim=torch.optim.SGD(model.parameters(),lr,momentum=0.9)
num_epochs=20
for i in range(num_epochs):
    model.train()
    running_loss=0.0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        loss=l_type(outputs,labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        running_loss+=loss.item()
    print(f"Epoch[{i+1}/{num_epochs}],Loss:{running_loss/len(train_loader):.4f}")    
model.eval()        
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

print(f"Test Accuracy:{100*correct/total:.2f}%")      


        