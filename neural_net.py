import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
data=datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test=datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(data, batch_size=64, shuffle=True)
test_loader=DataLoader(test, batch_size=64, shuffle=True)
train_iterator = iter(train_loader)
scaler=StandardScaler()
class NeuralNet(nn.Module):
    def __init__(self,input_dim,layer1_dim,layer2_dim):
        super().__init__()
        self.L1=nn.Linear(input_dim,layer1_dim)
        self.L2=nn.Linear(layer1_dim,layer2_dim)
        self.L3=nn.Linear(layer2_dim,10)
    def forward(self,X):
        X=F.relu(self.L1(X))
        X=F.relu(self.L2(X))
        X=self.L3(X)
        return X
input_dim=784
L1_dim=200
L2_dim=32
model=NeuralNet(input_dim,L1_dim,L2_dim)
L_type=nn.CrossEntropyLoss()
lr=0.01
p=0.9
optimizer=torch.optim.SGD(model.parameters(),lr,momentum=p)  
num_epochs=10
for epoch in range(num_epochs):
   model.train()
   running_loss=0.0
   for images,labels in train_loader:
        images=images.view(-1,784)
        outputs=model(images)
        loss=L_type(outputs,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss+=loss.item()
   print(f"Epoch[{epoch+1}/{num_epochs}],Loss:{running_loss/len(train_loader):.4f}")    
model.eval()
correct=0
total=0
with torch.no_grad():
    for images, labels in test_loader:
        images=images.view(-1,784)
        outputs=model(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
