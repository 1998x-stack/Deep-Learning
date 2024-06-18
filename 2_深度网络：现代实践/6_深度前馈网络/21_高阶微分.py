# 21_高阶微分


"""

Lecture: 2_深度网络：现代实践/6_深度前馈网络
Content: 21_高阶微分

"""


import torch

# 定义一个简单的函数
def f(x: torch.Tensor) -> torch.Tensor:
    
"""

    计算函数 f(x) = x^3 + 2x^2 + x 的值

    参数:
    x (torch.Tensor): 输入张量

    返回:
    torch.Tensor: 函数的值
    
"""

    return x**3 + 2*x**2 + x

# 使用自动微分计算一阶和二阶导数
x = torch.tensor([2.0], requires_grad=True)
y = f(x)

# 计算一阶导数
grad_1 = torch.autograd.grad(outputs=y, inputs=x, create_graph=True)[0]
print('一阶导数:', grad_1)

# 计算二阶导数
grad_2 = torch.autograd.grad(outputs=grad_1, inputs=x, retain_graph=True)[0]
print('二阶导数:', grad_2)
