# 00_实例：学习 XOR
"""
Lecture: 2_深度网络：现代实践/6_深度前馈网络
Content: 00_实例：学习 XOR
"""
## 00_实例：学习 XOR
### 任务分解：
1. **背景介绍**
2. **准备数据**
3. **构建神经网络模型**
4. **训练模型**
5. **评估模型**
6. **可视化结果**
### 1. 背景介绍
**步骤：**
- 解释 XOR 问题的定义及其在逻辑运算中的作用。
- 说明为什么 XOR 问题对于神经网络训练是一个经典且重要的实例。
**解释：**
XOR（异或）是一个简单的逻辑运算，它的输出只有在输入不同时才为真（1）。具体的真值表如下：
| 输入 A | 输入 B | 输出 A XOR B |
| ------ | ------ | ------------ |
|   0    |   0    |      0       |
|   0    |   1    |      1       |
|   1    |   0    |      1       |
|   1    |   1    |      0       |
XOR 问题是非线性可分的，这意味着无法通过简单的线性模型（如感知器）解决。因此，使用多层神经网络（如深度前馈网络）来解决这个问题是展示神经网络强大功能的一个经典案例。
### 2. 准备数据
**步骤：**
- 导入必要的库
- 创建 XOR 数据集
**代码：**
```python
import numpy as np
# XOR 输入和输出
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
```
### 3. 构建神经网络模型
**步骤：**
- 使用 PyTorch 构建一个简单的前馈神经网络
- 定义网络的层次结构
**代码：**
```python
import torch
import torch.nn as nn
import torch.optim as optim
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
```
### 4. 训练模型
**步骤：**
- 定义损失函数和优化器
- 训练模型并记录损失
**代码：**
```python
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
```
### 5. 评估模型
**步骤：**
- 计算模型在训练数据上的准确性
**代码：**
```python
# 评估模型
with torch.no_grad():
    predicted = model(torch.tensor(X, dtype=torch.float)).round()
    accuracy = (predicted.numpy() == y).mean()
    print(f'Accuracy: {accuracy * 100:.2f}%')
```
### 6. 可视化结果
**步骤：**
- 可视化训练损失的变化
- 展示模型在不同输入下的输出
**代码：**
```python
import matplotlib.pyplot as plt
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
plt.show()
```
通过以上步骤，我们完成了 XOR 问题的神经网络解决方案。每一步都详细解释了背后的逻辑和实现细节。