# Week 2 / Day 1：NumPy 基础语法

## 1. 导入 NumPy

标准写法：

```python
import numpy as np
```

`np` 是 NumPy 最常见的简称。之后使用：

```python
np.array(...)
np.zeros(...)
np.ones(...)
```

## 2. 创建 array

从 Python list 创建：

```python
vector = np.array([1, 2, 3])
```

创建二维 array：

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

指定数据类型：

```python
values = np.array([1, 2, 3], dtype=np.float32)
```

## 3. 查看 array 信息

```python
print(values)
print(values.shape)
print(values.ndim)
print(values.dtype)
print(values.size)
```

含义：

```text
shape：每个维度的长度。
ndim：维度数量。
dtype：元素类型。
size：所有元素的总数。
```

## 4. zeros、ones 和 arange

创建全是 0 的数组：

```python
zeros = np.zeros((2, 3))
```

创建全是 1 的数组：

```python
ones = np.ones((2, 3))
```

创建连续数字：

```python
numbers = np.arange(0, 6)
```

结果：

```text
[0 1 2 3 4 5]
```

## 5. 一维索引

```python
vector = np.array([10, 20, 30])

print(vector[0])   # 10
print(vector[-1])  # 30
```

与 Python list 一样，下标从 0 开始，`-1` 表示最后一个元素。

## 6. 二维索引

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

```python
print(matrix[0])     # 第一行
print(matrix[1, 2])  # 第二行第三列，也就是 6
print(matrix[:, 1])  # 所有行的第二列，也就是 [2 5]
```

`:` 在这里表示选取这个维度中的全部元素。

## 7. array 的逐元素运算

```python
values = np.array([1, 2, 3])

print(values + 10)
print(values * 2)
print(values ** 2)
```

输出：

```text
[11 12 13]
[2 4 6]
[1 4 9]
```

这里的运算会分别作用于每个元素。

## 8. 今日最重要的代码

```python
import numpy as np

samples = np.array([
    [60, 1, 20],
    [80, 2, 10],
    [100, 3, 5],
], dtype=np.float32)

print(samples.shape)
print(samples.ndim)
print(samples.dtype)
```

你应该能解释：

```text
为什么 shape 是 (3, 3)？
为什么 ndim 是 2？
为什么 dtype 是 float32？
```
