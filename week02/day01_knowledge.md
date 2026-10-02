# Week 2 / Day 1：NumPy Array、Shape 与 Dtype

Week 1 中，我们使用 Python list 保存数据：

```python
features = [80, 2, 10]
samples = [
    [60, 1, 20],
    [80, 2, 10],
    [100, 3, 5],
]
```

从今天开始，我们用 NumPy array 表示这些数据：

```python
import numpy as np

features = np.array([80, 2, 10])
```

NumPy 是 Python 数值计算的基础工具。以后 PyTorch 的 Tensor 与 NumPy array 在很多概念上非常相似。

## 1. 为什么学习 NumPy？

Python list 可以保存数据，但不擅长大规模数学运算。

例如，想让所有元素乘以 2：

```python
numbers = [1, 2, 3]
```

Python list 通常需要循环：

```python
results = []
for number in numbers:
    results.append(number * 2)
```

NumPy 可以直接计算：

```python
numbers = np.array([1, 2, 3])
results = numbers * 2
```

结果为：

```text
[2 4 6]
```

NumPy 的主要优点是：

```text
写法简洁
计算速度快
适合向量和矩阵
可以清楚记录数据形状
```

## 2. 什么是 array？

`array` 可以理解为专门用于数值计算的多维数据容器。

一维 array：

```python
vector = np.array([1, 2, 3])
```

二维 array：

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

三维 array 可以表示多批二维数据。以后 Transformer 中经常出现三维数据：

```text
[batch_size, sequence_length, d_model]
```

## 3. 什么是 shape？

`shape` 表示 array 每个维度的长度。

```python
vector = np.array([1, 2, 3])
print(vector.shape)
```

输出：

```text
(3,)
```

意思是一维数据，这个维度有 3 个元素。

二维例子：

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

print(matrix.shape)
```

输出：

```text
(2, 3)
```

含义是：

```text
2 行
每行 3 列
```

在机器学习数据中，也可以理解为：

```text
2 个样本
每个样本 3 个特征
```

## 4. 什么是 ndim？

`ndim` 表示 array 有多少个维度：

```python
vector.ndim  # 1
matrix.ndim  # 2
```

注意：维度数量和元素数量不是同一件事。

```text
[1, 2, 3, 4] 有 4 个元素，但它是 1 维。
[[1, 2], [3, 4]] 有 4 个元素，但它是 2 维。
```

## 5. 什么是 dtype？

`dtype` 表示 array 中元素的数据类型。

```python
integers = np.array([1, 2, 3])
floats = np.array([1.0, 2.0, 3.0])

print(integers.dtype)
print(floats.dtype)
```

常见类型包括：

```text
int64：整数
float64：双精度小数
float32：单精度小数，深度学习中非常常见
bool：布尔值 True / False
```

可以主动指定类型：

```python
values = np.array([1, 2, 3], dtype=np.float32)
```

## 6. Shape 与 Transformer 的关系

Transformer 中经常需要追踪：

```text
(batch_size, sequence_length, d_model)
```

例如：

```text
(32, 10, 512)
```

表示：

```text
一批有 32 个样本
每个样本有 10 个 token
每个 token 用 512 个数字表示
```

Transformer 最常见的错误之一就是 shape 不匹配。因此，从今天开始，每看到一份数据都要习惯问：

```text
它有几个维度？
每个维度分别代表什么？
它的 shape 是什么？
```

## 7. 今天要形成的直觉

```text
array：用于数值计算的数据容器。
shape：每个维度分别有多长。
ndim：一共有几个维度。
dtype：里面的数字是什么类型。
```

不要只背 `(2, 3)`，要能解释它在当前问题里代表“2 个样本，每个样本 3 个特征”。

## 8. 今日关键词

```text
NumPy
array：数组
shape：形状
dimension：维度
ndim：维度数量
dtype：数据类型
vector：向量
matrix：矩阵
```
