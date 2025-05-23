import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
class LogisticRegression(nn.Module):
   def __init__(self,input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
   def forward(self, x):
        return torch.sigmoid(self.linear(x))
data=load_breast_cancer()
X,y=data.data,data.target.reshape(-1,1)
scaler=StandardScaler()
X =scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train=torch.from_numpy(X_train.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))
input_dim=X_train.shape[1]
model=LogisticRegression(input_dim)
lr=0.01
loss_type=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr)

num_epochs=150
for i in range(num_epochs):
    y_cap=model(X_train)
    loss=loss_type(y_cap,y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (i+1)%10==0:
        print(f"Epoch {i+1}: Loss = {loss.item():.4f}")
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = y_pred.round()  # Threshold at 0.5
    acc = (y_pred_class.eq(y_test).sum().item()) / y_test.shape[0]
    print(f"Test Accuracy: {(acc*100):.2f}%")

