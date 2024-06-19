# 05_整流线性单元及其扩展
"""
Lecture: 2_深度网络：现代实践/6_深度前馈网络
Content: 05_整流线性单元及其扩展
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, activation='relu'):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        self.fc2 = nn.Linear(2, 1)
        self.output = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return self.output(x)
# 初始化模型
model = SimpleNN(activation='leaky_relu')
# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 准备数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
# 训练模型
epochs = 10000
losses = []
for epoch in range(epochs):
    # 前向传播
    outputs = model(torch.tensor(X, dtype=torch.float))
    loss = criterion(outputs, torch.tensor(y, dtype=torch.float))
    losses.append(loss.item())
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
# 评估模型
with torch.no_grad():
    predicted = model(torch.tensor(X, dtype=torch.float)).round()
    accuracy = (predicted.numpy() == y).mean()
    print(f'Accuracy: {accuracy * 100:.2f}%')
