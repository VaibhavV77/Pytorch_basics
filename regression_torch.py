import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
data=load_diabetes()
X, y = data.data, data.target.reshape(-1, 1)
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1) 
input_sz=X.shape[1]
output_sz=1
model=nn.Linear(input_sz,output_sz)
l_type=nn.MSELoss()
lr=0.01
opt=torch.optim.SGD(model.parameters(),lr)
NE=X.shape[0]
for epochs in range(NE):
    y_cap=model(X)
    loss=l_type(y_cap,y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    if (epochs+1)%10==0:
        print(f"Epoch {epochs+1}: Loss = {loss.item():.4f}")
y_pred = model(X).detach().numpy()
y_true = y.numpy()
y_pred = y_scaler.inverse_transform(y_pred)
y_true = y_scaler.inverse_transform(y_true)
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # ideal line
plt.xlabel("Actual Disease Progression")
plt.ylabel("Predicted Progression")
plt.title("Actual vs. Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.show()