# Week 2 / Day 3：点积与矩阵乘法

Week 1 中，我们用循环计算多特征线性模型：

```python
total = 0

for feature, weight in zip(features, weights):
    total += feature * weight

prediction = total + bias
```

今天会发现，这段代码本质上就是点积：

```python
prediction = features @ weights + bias
```

## 1. 什么是点积？

给定两个长度相同的向量：

```text
x = [1, 2, 3]
w = [4, 5, 6]
```

点积的计算是对应元素相乘后求和：

```text
x · w = 1*4 + 2*5 + 3*6
      = 4 + 10 + 18
      = 32
```

用 NumPy 写：

```python
x = np.array([1, 2, 3])
w = np.array([4, 5, 6])

result = x @ w
```

结果是一个数：

```text
32
```

## 2. 为什么点积就是线性模型？

多特征线性模型：

```text
y = x1*w1 + x2*w2 + x3*w3 + b
```

把特征和权重写成向量：

```text
x = [x1, x2, x3]
w = [w1, w2, w3]
```

公式就可以简写为：

```text
y = x · w + b
```

NumPy 实现：

```python
y = x @ w + b
```

## 3. 一批样本如何同时预测？

假设有 3 个样本，每个样本有 2 个特征：

```python
samples = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])
```

shape 是：

```text
(3, 2)
```

表示 3 个样本、每个样本 2 个特征。

权重：

```python
weights = np.array([10, 20])
```

shape 是 `(2,)`。

批量预测：

```python
predictions = samples @ weights
```

shape 变化：

```text
(3, 2) @ (2,) -> (3,)
```

结果：

```text
[ 50 110 170]
```

它等价于分别计算：

```text
1*10 + 2*20 = 50
3*10 + 4*20 = 110
5*10 + 6*20 = 170
```

NumPy 一次完成了整批数据的计算。

## 4. 什么是矩阵乘法？

当左右两边都是二维 array 时，`@` 表示矩阵乘法。

```text
A.shape = (2, 3)
B.shape = (3, 4)
```

可以计算：

```text
A @ B
```

结果的 shape 是：

```text
(2, 4)
```

可以用这条规则判断：

```text
(2, 3) @ (3, 4) -> (2, 4)
    ^       ^
    中间两个维度必须相同
```

也就是：

```text
(m, n) @ (n, p) -> (m, p)
```

内侧的 `n` 必须相同，结果保留外侧的 `m` 和 `p`。

## 5. 为什么神经网络需要矩阵乘法？

假设输入数据：

```text
X.shape = (batch_size, input_features)
```

一层神经网络的权重：

```text
W.shape = (input_features, output_features)
```

那么：

```text
X @ W -> (batch_size, output_features)
```

再加偏置：

```text
Y = X @ W + b
```

这正是神经网络线性层的核心公式。

## 6. 多输出线性层

一个输入可以同时产生多个输出。

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

`X.shape = (2, 3)`：2 个样本，每个样本 3 个输入特征。

```python
W = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])
```

`W.shape = (3, 2)`：把 3 个输入特征转换成 2 个输出特征。

```python
Y = X @ W
```

shape：

```text
(2, 3) @ (3, 2) -> (2, 2)
```

也就是每个样本得到 2 个输出。

## 7. `*` 和 `@` 不一样

NumPy 中：

```text
*：对应位置逐元素相乘
@：点积或矩阵乘法
```

例如：

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
```

```text
a * b = [4, 10, 18]
a @ b = 32
```

不要混淆这两个运算。

## 8. 与 Transformer 的关系

Transformer 中大量核心计算都是矩阵乘法：

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv

Attention scores = Q @ K.T
Attention output = weights @ V
```

因此，能熟练判断矩阵乘法前后的 shape，是以后理解 Attention 的基础。

## 9. 今日关键词

```text
dot product：点积
element-wise multiplication：逐元素乘法
matrix multiplication：矩阵乘法
input features：输入特征
output features：输出特征
```
