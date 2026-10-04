# Week 3 / Day 1：神经元与线性层

前两周学习的多特征线性模型是：

```text
y = x1*w1 + x2*w2 + ... + b
```

在神经网络中，这正是一个神经元最核心的计算。

## 1. 什么是神经元？

一个神经元接收多个输入，为每个输入准备一个权重，把它们加权求和，再加上偏置：

```text
z = x1*w1 + x2*w2 + ... + b
```

向量形式：

```text
z = x @ w + b
```

NumPy 实现：

```python
z = inputs @ weights + bias
```

```text
inputs：一个样本的所有输入特征
weights：神经元对每个特征的权重
bias：神经元的偏置
z：神经元的线性输出
```

## 2. 神经元和线性回归的关系

如果只看线性计算，它们没有本质区别：

```text
线性回归：prediction = x @ w + b
神经元：   z          = x @ w + b
```

神经网络通常会把 `z` 送入激活函数，再连接更多层：

```text
线性计算 z = x @ w + b
        ↓
激活函数 a = activation(z)
```

今天只学习线性计算，下一天学习激活函数。

## 3. 多个神经元组成线性层

假设输入有 3 个特征，我们希望同时得到 2 个输出，就可以放置 2 个神经元。

把两个神经元的权重放进矩阵：

```text
W = [[w11, w12],
     [w21, w22],
     [w31, w32]]
```

```text
W.shape = (3, 2)
```

其中：

```text
每一行对应一个输入特征
每一列对应一个输出神经元
```

两个神经元各有一个偏置：

```text
b.shape = (2,)
```

## 4. 一批样本经过线性层

假设有 4 个样本，每个样本有 3 个特征：

```text
X.shape = (4, 3)
W.shape = (3, 2)
b.shape = (2,)
```

计算：

```python
Z = X @ W + b
```

shape：

```text
(4, 3) @ (3, 2) + (2,) -> (4, 2)
```

结果表示 4 个样本分别经过 2 个神经元，每个样本得到 2 个输出。

## 5. 线性层的通用 Shape

```text
X.shape = (batch_size, in_features)
W.shape = (in_features, out_features)
b.shape = (out_features,)
Z.shape = (batch_size, out_features)
```

公式：

```text
Z = X @ W + b
```

`out_features` 就是这一层的神经元数量。

## 6. 权重初始化

创建线性层时，需要给权重初始值：

```python
rng = np.random.default_rng(seed=42)
weights = rng.normal(
    0.0,
    0.1,
    size=(in_features, out_features),
).astype(np.float32)
```

它会生成均值为 0、标准差为 0.1 的小随机数。指定相同 `seed` 可以复现实验结果。

偏置可以初始化为 0：

```python
bias = np.zeros(out_features, dtype=np.float32)
```

## 7. 用 Class 表示线性层

```python
class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weights = ...
        self.bias = ...

    def forward(self, X):
        return X @ self.weights + self.bias
```

这里使用 `forward()` 而不是 `predict()`，因为神经网络中的每一层都在执行一次前向计算。

## 8. 与 PyTorch 和 Transformer 的关系

PyTorch 中会写：

```python
nn.Linear(in_features=3, out_features=2)
```

Transformer 中也有大量线性层：

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv
Feed Forward Network 中的两层 Linear
Multi-Head Attention 最后的输出投影
```

今天实现的 `LinearLayer` 是这些模块的基础零件。

## 9. 今日关键词

```text
neuron：神经元
linear layer：线性层
in_features：输入特征数量
out_features：输出特征数量，也就是神经元数量
forward：前向传播
weight matrix：权重矩阵
bias vector：偏置向量
```
