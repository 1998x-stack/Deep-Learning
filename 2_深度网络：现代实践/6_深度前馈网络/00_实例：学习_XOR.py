# 00_实例：学习 XOR
"""
Lecture: 2_深度网络：现代实践/6_深度前馈网络
Content: 00_实例：学习 XOR
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# XOR 输入和输出
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
# 定义神经网络模型
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
model = XORModel()
# 损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 训练模型
epochs = 10000
for epoch in range(epochs):
    # 前向传播
    outputs = model(torch.tensor(X, dtype=torch.float))
    loss = criterion(outputs, torch.tensor(y, dtype=torch.float))
    
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
# 可视化训练损失
losses = []
for epoch in range(epochs):
    outputs = model(torch.tensor(X, dtype=torch.float))
    loss = criterion(outputs, torch.tensor(y, dtype=torch.float))
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('figures/00_实例：学习 XOR.png')
plt.close()
