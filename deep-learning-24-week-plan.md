# 从零开始学习深度学习：24 周长期项目计划

## 项目目标

这个长期项目的目标不是简单学会调用 PyTorch 或现成的 Transformer API，而是从零开始建立深度学习能力，最终能够：

- 理解神经网络的基本原理
- 看懂基础深度学习论文中的网络结构和公式
- 使用 PyTorch 自己实现模型
- 训练、调参、测试和 debug 模型
- 阅读 GitHub 上别人的深度学习项目
- 从零实现 Transformer，并完成一个可展示的完整 GitHub 项目

最终希望从：

```python
我不知道神经网络怎么训练
```

逐渐变成：

```python
class MultiHeadAttention(nn.Module):
    ...

class EncoderLayer(nn.Module):
    ...

class Transformer(nn.Module):
    ...
```

重点不是复制代码，而是知道每个模块为什么这样写、数据 shape 如何变化、梯度如何更新参数。

## 总体路线

学习路线分为 7 个阶段：

```text
Stage 0   Python + NumPy
              ↓
Stage 1   深度学习数学
              ↓
Stage 2   神经网络基本原理
              ↓
Stage 3   PyTorch
              ↓
Stage 4   CNN / RNN / LSTM 等经典网络
              ↓
Stage 5   Attention + Transformer
              ↓
Stage 6   论文阅读、复现与 GitHub 项目
```

核心原则：

> 不把“学理论”和“写代码”分开。

例如学习 Linear Layer 时，要同时理解：

```text
数学：y = Wx + b
理论：Linear Layer 是什么
代码：nn.Linear()
手写：自己用矩阵乘法实现 Linear
实验：训练一个小模型
```

## 学习工作模式

后续每天学习按照这个节奏进行：

```text
知识讲解
→ 示例代码
→ 小练习
→ 当天作业
→ GitHub 项目积累
→ 根据掌握情况调整进度
```

每学一个模型，都问自己 5 个问题：

1. 它解决什么问题？
2. 输入和输出是什么？
3. 数据经过模型时 shape 怎么变化？
4. Loss 怎么计算？
5. Gradient 怎么回去更新参数？

尤其要从一开始建立 shape 意识。例如 Transformer 中常见的数据变化：

```text
Input
[B, T]

↓

Embedding
[B, T, D]

↓

Q / K / V
[B, H, T, Dh]

↓

Attention
[B, H, T, T]

↓

Output
[B, T, D]
```

## Stage 0：Python + NumPy

时间：1-2 周

你有 C/C++ 编程基础，所以 Python 不需要学很久，重点学习深度学习会用到的部分。

### Python 重点

```text
变量
list / tuple / dict
if / for / while
函数
lambda
class
继承
import
模块
文件操作
virtual environment
pip
```

尤其要熟悉：

```python
class Model:
    def __init__(self):
        ...
```

因为之后 PyTorch 中会经常写：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
```

### NumPy 重点

```python
import numpy as np

np.array
np.zeros
np.ones
np.random

reshape
transpose
squeeze
concatenate

sum
mean
max
argmax

矩阵乘法
广播 broadcasting
```

第一个小项目：

> 用 NumPy 从零实现 Linear Regression。

目标是理解：

```text
y_hat = wx + b

Loss = 1/N * sum((y_hat - y)^2)
```

并自己实现 gradient descent。

## Stage 1：深度学习数学

时间：2-3 周，可以和 PyTorch 同时学。

不需要把高数、线性代数、概率论整本重新学一遍，只学习深度学习真正需要的数学。

### 线性代数

重点：

```text
向量
矩阵
矩阵乘法
transpose
inverse
dot product
norm
线性变换
eigenvalue
eigenvector
```

最重要的是矩阵乘法和维度判断。

例如：

```text
X: [32, 512]
W: [512, 256]

XW: [32, 256]
```

以后 Transformer 里的：

```text
Q = XW_Q
K = XW_K
V = XW_V
```

都依赖这些基础。

### 微积分与梯度

重点：

```text
导数
偏导数
链式法则
梯度
梯度下降
```

必须理解：

```text
loss.backward()
```

背后到底做了什么。

## Stage 2：神经网络与反向传播

时间：3-4 周

这一阶段非常关键，不要急着进入 Transformer。

需要理解：

```text
Neuron
Linear Layer
Activation Function
Loss Function
Forward Propagation
Backpropagation
Gradient Descent
Optimizer
```

第一个神经元：

```text
z = w1*x1 + w2*x2 + b
a = sigmoid(z)
```

多个 neuron：

```text
z = Wx + b
```

你会逐渐发现：

> 神经网络主要就是矩阵运算 + 非线性函数。

### Activation Function

重点学习：

```text
Sigmoid
Tanh
ReLU
GELU
Softmax
```

Transformer 后面会经常遇到：

```text
ReLU
GELU
Softmax
```

### Loss Function

重点掌握：

```text
MSE Loss
Binary Cross Entropy
Cross Entropy
```

尤其是：

> Cross Entropy + Softmax

它们在分类任务和 Transformer 中都非常重要。

### Backpropagation

这是入门过程中最值得认真学的一部分。

例如：

```text
y = wx
L = (y - t)^2
```

需要理解：

```text
dL / dw
```

如何通过 chain rule 算出来。

之后扩展到：

```text
Linear
↓
ReLU
↓
Linear
↓
Softmax
↓
Loss
```

理解梯度如何一层层向前传播回去。

## Stage 3：PyTorch

时间：3 周左右

正式开始系统使用 PyTorch。

首先掌握 Tensor：

```python
import torch

x = torch.tensor(...)
x.shape

x.reshape(...)
x.transpose(...)

x @ y
```

然后学习：

```text
Tensor
Autograd
Dataset
DataLoader
nn.Module
Optimizer
GPU
```

重点理解 autograd：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()

print(x.grad)
```

运行前应该能知道：

```text
x.grad = 4
```

因为：

```text
dy/dx = 2x
x = 2
dy/dx = 4
```

### 第一个 PyTorch 神经网络

例如 MNIST：

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

你必须能解释每一行。

例如：

```python
nn.Linear(784, 256)
```

实际上对应：

```text
XW + b
```

而不是把 `nn.Linear` 当黑箱。

## Stage 4：CNN / RNN / LSTM

时间：3-4 周

这一阶段开始接触经典深度学习模型。

学习顺序：

```text
MLP
↓
CNN
↓
RNN
↓
LSTM
↓
Attention
```

### CNN

至少学习：

```text
Convolution
Kernel
Stride
Padding
Pooling
Feature Map
```

项目：

```text
Project 1：MNIST 手写数字分类
Project 2：CIFAR-10 图片分类
```

典型结构：

```text
Conv
↓
ReLU
↓
Pooling
↓
Conv
↓
ReLU
↓
Pooling
↓
Linear
↓
Classification
```

### RNN / LSTM

因为最终目标更偏 Transformer，所以 RNN / LSTM 不需要学得特别深。

需要知道：

```text
RNN 是什么
Hidden State 是什么
RNN 为什么有 long-term dependency 问题
Vanishing Gradient 是什么
LSTM 如何缓解
为什么 Transformer 可以不用 recurrence
```

## Stage 5：Attention

这是进入 Transformer 的核心阶段。

必须理解：

```text
Q
K
V
```

以及公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

不要背公式，要逐项搞明白：

```text
Q 是什么？
K 是什么？
为什么 QK^T？
为什么除以 sqrt(d_k)？
为什么 Softmax？
为什么最后乘 V？
```

如果这些问题都能自己解释，Transformer 最难的一关基本就过去了。

## Stage 6：Transformer、论文阅读与复现

这时再读经典论文：

> Attention Is All You Need

论文地址：

https://arxiv.org/abs/1706.03762

不要第一次就试图逐字读完。

建议读三遍：

```text
第一次：看整体结构
Encoder / Decoder
Self Attention
Multi Head Attention
Feed Forward

第二次：看公式

第三次：结合代码
```

最终要自己实现这些模块：

```text
Embedding
↓
Positional Encoding
↓
Scaled Dot Product Attention
↓
Multi Head Attention
↓
Feed Forward Network
↓
LayerNorm
↓
Residual Connection
↓
Encoder Layer
↓
Encoder
↓
Decoder Layer
↓
Decoder
↓
Transformer
```

最后代码中应该出现：

```python
class MultiHeadAttention(nn.Module):
    ...

class FeedForward(nn.Module):
    ...

class EncoderLayer(nn.Module):
    ...

class DecoderLayer(nn.Module):
    ...

class Transformer(nn.Module):
    ...
```

## 训练能力

不要只写模型，也要学会训练。

必须掌握：

```text
Train / Validation / Test
Batch
Epoch
Learning Rate
Optimizer
SGD
Adam
AdamW
Weight Decay
Dropout
Overfitting
Underfitting
Learning Rate Scheduler
Gradient Clipping
Checkpoint
```

最终应该能够看到：

```text
training loss 不下降
```

然后自己分析：

```text
数据有问题？
模型有问题？
learning rate 有问题？
gradient 有问题？
loss 写错？
shape 错了？
mask 错了？
```

这才是真正开始具备做项目和科研的能力。

## 24 周安排

建议每周投入 8-12 小时，不需要为了赶进度每天硬学五六小时。

| 时间 | 内容 | 项目 |
|---|---|---|
| Week 1-2 | Python + NumPy | Linear Regression |
| Week 3-4 | 线性代数 + 微积分 | 手算 Gradient |
| Week 5-6 | Neural Network | NumPy MLP |
| Week 7-8 | PyTorch | PyTorch MLP |
| Week 9-10 | CNN | MNIST |
| Week 11-12 | CNN 深入 | CIFAR-10 |
| Week 13 | RNN | Sequence Model |
| Week 14 | LSTM | Text Classification |
| Week 15-16 | Attention | 手写 Attention |
| Week 17-18 | Transformer | Encoder |
| Week 19-20 | Transformer | 完整 Transformer |
| Week 21 | 读论文 | Attention Is All You Need |
| Week 22 | 复现 | Transformer |
| Week 23 | 实验 | 调参 + Ablation |
| Week 24 | GitHub | 整理完整项目 |

## GitHub 项目积累

不要等学完才建 GitHub，从第一周就开始。

建议仓库名：

```text
deep-learning-from-scratch
```

最终结构可以逐渐形成：

```text
deep-learning-from-scratch/

├── 01_numpy/
│   ├── linear_regression.py
│   └── matrix_operations.py
│
├── 02_neural_network/
│   ├── neuron.py
│   ├── activation.py
│   └── mlp_numpy.py
│
├── 03_pytorch/
│   └── mlp.py
│
├── 04_cnn/
│   ├── mnist.py
│   └── cifar10.py
│
├── 05_rnn/
│
├── 06_attention/
│   └── attention.py
│
└── 07_transformer/
    ├── attention.py
    ├── encoder.py
    ├── decoder.py
    ├── transformer.py
    └── train.py
```

半年以后，目标不是只说：

> 我学过深度学习。

而是拥有一个完整的：

> Deep Learning From Scratch 项目。

## 当前起点：Week 1 / Day 1

从 Week 1 / Day 1 开始，第一天目标是建立 Python 基础和线性模型意识。

今天掌握：

```text
变量
list
for 循环
函数
简单 class
用代码表达数学计算
```

深度学习代码经常在做：

```text
数据 → 模型 → 预测结果 → 计算误差 → 更新参数
```

最简单的模型：

```text
y = w*x + b
```

其中：

```text
x 是输入
w 是权重
b 是偏置
y 是输出
```

这就是以后神经网络中 Linear Layer 的雏形。

### Day 1 小练习

写代码完成：

```text
输入 xs = [1, 2, 3, 4, 5, 6]
使用公式 y = 3x + 2
打印每一个 x 对应的 y
```

期望输出：

```text
1 5
2 8
3 11
4 14
5 17
6 20
```

### Day 1 作业

任务 1：写一个函数：

```python
def predict(x):
    ...
```

让它实现：

```text
y = 4x - 1
```

任务 2：用这个函数计算：

```python
xs = [2, 4, 6, 8, 10]
```

任务 3：写一个简单 class：

```python
class LinearModel:
    def __init__(self, w, b):
        ...

    def predict(self, x):
        ...
```

让它可以这样使用：

```python
model = LinearModel(4, -1)
print(model.predict(2))
```

### Day 1 GitHub 积累

建议新增：

```text
week01/
  day01_python_linear_model.py
  notes.md
```

`notes.md` 里写：

```text
今天学了什么？
哪里不懂？
线性模型 y = wx + b 是什么意思？
```

## 后续调整原则

这份计划不是死板课程表。

如果某一部分掌握得快，就提前进入下一阶段；如果反向传播、shape、PyTorch 训练循环这些关键点没有真正理解，就放慢速度，多做小练习和小项目。

最终标准不是“看完了多少内容”，而是：

- 能不能自己写出来
- 能不能解释每一步
- 能不能根据 loss 和 shape debug
- 能不能把代码整理成一个别人能看懂的 GitHub 项目
