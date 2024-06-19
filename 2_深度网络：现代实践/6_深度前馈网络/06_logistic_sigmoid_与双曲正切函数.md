# 06_logistic sigmoid 与双曲正切函数
"""
Lecture: 2_深度网络：现代实践/6_深度前馈网络
Content: 06_logistic sigmoid 与双曲正切函数
"""
## 06_logistic sigmoid 与双曲正切函数
### 任务分解：
1. **背景介绍**
2. **Logistic Sigmoid 函数的定义**
3. **双曲正切函数（Tanh）的定义**
4. **Logistic Sigmoid 与 Tanh 的优缺点**
5. **Logistic Sigmoid 与 Tanh 的应用场景**
6. **实现 Logistic Sigmoid 和 Tanh 函数**
7. **训练和评估模型**
8. **可视化 Logistic Sigmoid 与 Tanh 函数**
### 1. 背景介绍
**步骤：**
- 解释激活函数在神经网络中的作用。
- 强调 Logistic Sigmoid 和 Tanh 函数的重要性及其应用场景。
**解释：**
激活函数是神经网络中的关键组件，用于引入非线性，使网络能够学习复杂的模式和特征。Logistic Sigmoid 和双曲正切（Tanh）函数是两种常用的激活函数，广泛应用于早期的神经网络模型中。
### 2. Logistic Sigmoid 函数的定义
**步骤：**
- 提供 Logistic Sigmoid 函数的数学定义。
- 说明 Logistic Sigmoid 如何在神经网络中进行信息处理。
**解释：**
Logistic Sigmoid 函数的数学定义为：
$$ f(x) = \frac{1}{1 + e^{-x}} $$
它将输入值压缩到 [0, 1] 之间，使得输出可以被解释为概率值。
### 3. 双曲正切函数（Tanh）的定义
**步骤：**
- 提供双曲正切函数（Tanh）的数学定义。
- 说明 Tanh 如何在神经网络中进行信息处理。
**解释：**
双曲正切（Tanh）函数的数学定义为：
$$ f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
它将输入值压缩到 [-1, 1] 之间，相对于 Logistic Sigmoid，Tanh 在输出上具有零中心性。
### 4. Logistic Sigmoid 与 Tanh 的优缺点
**步骤：**
- 介绍 Logistic Sigmoid 和 Tanh 的主要优点。
- 讨论 Logistic Sigmoid 和 Tanh 的潜在缺点。
**解释：**
- **Logistic Sigmoid 优点**：
  - 简单易计算
  - 输出值在 [0, 1] 之间，易于解释为概率
  
- **Logistic Sigmoid 缺点**：
  - 梯度消失问题：在输入绝对值较大时，梯度接近于零，导致训练速度变慢
  
- **Tanh 优点**：
  - 输出范围为 [-1, 1]，具有零中心性，更有利于梯度下降
  - 在某些情况下表现优于 Logistic Sigmoid
  
- **Tanh 缺点**：
  - 同样存在梯度消失问题，特别是在深层网络中
### 5. Logistic Sigmoid 与 Tanh 的应用场景
**步骤：**
- 讨论 Logistic Sigmoid 和 Tanh 的适用场景。
- 说明何时选择使用 Logistic Sigmoid 或 Tanh。
**解释：**
- Logistic Sigmoid 适用于输出需要表示为概率的场景，如二分类问题的输出层。
- Tanh 适用于隐藏层，尤其是在需要零中心化输出的情况下，相比 Logistic Sigmoid 更能帮助网络更快地收敛。
### 6. 实现 Logistic Sigmoid 和 Tanh 函数
**步骤：**
- 使用 PyTorch 实现 Logistic Sigmoid 和 Tanh 函数。
- 演示如何在神经网络中使用这些激活函数。
**代码：**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, activation='sigmoid'):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        self.fc2 = nn.Linear(2, 1)
        self.output = nn.Sigmoid()  # 假设是一个二分类问题
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return self.output(x)
# 初始化模型
model = SimpleNN(activation='tanh')
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
```
### 7. 训练和评估模型
**步骤：**
- 训练模型并记录损失。
- 评估模型在训练数据上的准确性。
**代码：**
```python
# 训练和评估模型的代码已包含在上一步中
```
### 8. 可视化 Logistic Sigmoid 与 Tanh 函数
**步骤：**
- 可视化不同激活函数的输出。
- 展示不同激活函数在训练过程中的表现差异。
**代码：**
```python
import matplotlib.pyplot as plt
# 可视化训练损失
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss with Tanh Activation Function')
plt.show()
```
