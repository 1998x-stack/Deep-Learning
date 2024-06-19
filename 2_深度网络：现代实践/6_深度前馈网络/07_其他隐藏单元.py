# 07_其他隐藏单元
"""
Lecture: 2_深度网络：现代实践/6_深度前馈网络
Content: 07_其他隐藏单元
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# 定义 Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
# 定义 Mish 激活函数
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))
# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, activation='swish'):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        if activation == 'swish':
            self.activation = Swish()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'mish':
            self.activation = Mish()
        self.fc2 = nn.Linear(2, 1)
        self.output = nn.Sigmoid()  # 假设是一个二分类问题
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return self.output(x)
# 初始化模型
model = SimpleNN(activation='mish')
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