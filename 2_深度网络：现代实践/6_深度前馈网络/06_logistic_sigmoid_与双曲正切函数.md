# Sigmoid 与 Tanh：饱和、梯度和零中心

> 两个经典 S 形函数，是理解梯度消失的入口。

Sigmoid 与 tanh 都把输入压缩到有限区间；它们的输出范围、导数和适用位置各不相同。

网页交互版：[Sigmoid 与 Tanh：饱和、梯度和零中心](https://1998x-stack.github.io/Deep-Learning/articles/sigmoid-tanh/)

证据账本：[evidence.json](../../../site/assets/data/sigmoid-tanh/evidence.json)

## 学习目标

- 解释核心机制及适用边界。
- 把公式与计算图、数值示例对应起来。
- 通过独立 SVG 图示复核关键步骤。
- 使用配套 Python 例子核对核心计算。

## 1. 1. 输出范围不同

sigmoid 将实数映射到 (0,1)，适合表示 Bernoulli 概率；tanh 映射到 (−1,1)，输出以 0 为中心。

$$
σ(z)∈(0,1),tanh(z)∈(−1,1)
$$

![图 1：1. 输出范围不同](../../../site/assets/svg/sigmoid-tanh/fig-01.svg)

**图解：** 图 1. tanh 跨越 0，sigmoid 位于 0 到 1。

## 2. 2. 导数在中心附近最大

sigmoid′=σ(1−σ)，最大值 1/4；tanh′=1−tanh²，最大值 1。两者在 z=0 附近最敏感。

$$
σ′=σ(1−σ)tanh′=1−tanh2
$$

![图 2：2. 导数在中心附近最大](../../../site/assets/svg/sigmoid-tanh/fig-02.svg)

**图解：** 图 2. 导数在中心大，两端趋近于零。

## 3. 3. 饱和区削弱梯度

输入绝对值很大时，sigmoid 和 tanh 的曲线趋近平坦，导数接近零。多层链式相乘可能使靠近输入的层得到很弱梯度；权重尺度与架构也同样重要。

$$
∂h∂x=∏l=1Lφ′(zl)Wl
$$

![图 3：3. 饱和区削弱梯度](../../../site/assets/svg/sigmoid-tanh/fig-03.svg)

**图解：** 图 3. 输出层 sigmoid 有清楚的概率解释。

## 4. 4. Tanh 是缩放平移后的 sigmoid

两者满足 tanh(z)=2σ(2z)−1。tanh 的零中心输出可能让后续层接收正负都有的表示，但不能据此保证所有模型都收敛更快。

$$
tanh(z)=2σ(2z)−1
$$

![图 4：4. Tanh 是缩放平移后的 sigmoid](../../../site/assets/svg/sigmoid-tanh/fig-04.svg)

**图解：** 图 4. 饱和会降低局部梯度幅度。

## 5. 5. Sigmoid 常用于二分类输出

输出层 sigmoid 将一个 logit 转为 Bernoulli 正类概率。隐藏层 sigmoid 则不一定有概率语义，深层饱和可能削弱梯度。

$$
p(y=1|x)=σ(a)
$$

![图 5：5. Sigmoid 常用于二分类输出](../../../site/assets/svg/sigmoid-tanh/fig-05.svg)

**图解：** 图 5. tanh 输出有正有负。

## 6. 6. Tanh 仍适合有界正负状态

tanh 常用于需要有界状态的结构，例如部分循环网络候选状态。它输出正负值，但在两端同样会饱和。

$$
h=tanh(a)∈(−1,1)
$$

![图 6：6. Tanh 仍适合有界正负状态](../../../site/assets/svg/sigmoid-tanh/fig-06.svg)

**图解：** 图 6. 梯度消失是多因素共同作用。

## 7. 7. 初始化影响饱和概率

权重尺度过大时，预激活可能频繁落在饱和区。Glorot/Xavier 初始化旨在让层间信号尺度更合理，帮助缓解深层网络训练困难。

$$
Var(W)=2fanin+fanout
$$

![图 7：7. 初始化影响饱和概率](../../../site/assets/svg/sigmoid-tanh/fig-07.svg)

**图解：** 图 7. 函数放在不同层，语义也不同。

## 8. 8. 选择时同时看语义与梯度

需要概率时 sigmoid 有自然解释；需要有界、零中心状态时可考虑 tanh；深层隐藏层通常先比较 ReLU 家族。最终要结合初始化、输入尺度和验证集。

$$
选择={范围,导数,位置,初始化}
$$

![图 8：8. 选择时同时看语义与梯度](../../../site/assets/svg/sigmoid-tanh/fig-08.svg)

**图解：** 图 8. 激活需要与初始化和输入尺度一起考虑。

## 可运行示例

网页版本提供交互实验。下面代码展示核心计算；完整独立脚本位于本目录对应 Python 文件。

~~~python
import torch
z=torch.tensor([-6.,-2.,0.,2.,6.])
print(torch.sigmoid(z))
print(torch.tanh(z))
~~~

## 参考资料

- [输出范围不同](#s1)
- [导数在中心附近最大](#s2)
- [饱和区削弱梯度](#s3)
- [Tanh 是缩放平移后的 sigmoid](#s4)
- [Sigmoid 常用于二分类输出](#s5)
- [Tanh 仍适合有界正负状态](#s6)
- [初始化影响饱和概率](#s7)
- [选择时同时看语义与梯度](#s8)
- [Deep Learning, Goodfellow et al., Chapter 6.3.2](https://www.deeplearningbook.org/contents/mlp.html)
- [Glorot & Bengio, AISTATS 2010](https://proceedings.mlr.press/v9/glorot10a)

## 证据边界

教材事实、原始论文结果和本页推导分别记录在 evidence.json 中。教学示例不等同于完整基准复现；不要把单一示例外推为普遍性能结论。
