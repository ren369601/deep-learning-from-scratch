# Week 2 / Day 3：点积与矩阵乘法语法

## 1. 手写点积

```python
def manual_dot(a, b):
    total = 0

    for value_a, value_b in zip(a, b):
        total += value_a * value_b

    return total
```

这就是 Week 1 学过的“对应元素相乘后求和”。

## 2. 使用 `@`

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = a @ b
```

结果是 `32`。

## 3. 使用 `np.dot()`

一维向量时，下面两种写法效果相同：

```python
a @ b
np.dot(a, b)
```

本课程后面主要使用 `@`，因为矩阵乘法的含义更直观。

## 4. 逐元素乘法

```python
elementwise = a * b
```

结果：

```text
[4, 10, 18]
```

如果还要求和：

```python
dot_result = (a * b).sum()
```

结果也是 `32`。

## 5. 矩阵乘向量

```python
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])
w = np.array([10, 20])

y = X @ w
```

shape：

```text
(3, 2) @ (2,) -> (3,)
```

## 6. 矩阵乘矩阵

```python
A = np.ones((2, 3))
B = np.ones((3, 4))
C = A @ B
```

shape：

```text
(2, 3) @ (3, 4) -> (2, 4)
```

## 7. 线性层公式

```python
output = inputs @ weights + bias
```

典型 shape：

```text
inputs:  (batch_size, input_features)
weights: (input_features, output_features)
bias:    (output_features,)
output:  (batch_size, output_features)
```

`bias` 会通过广播加到每个样本的输出上。

## 8. 计算前先写 shape

遇到矩阵乘法，先写：

```text
(m, n) @ (n, p) -> (m, p)
```

再对照代码中的 shape。这个习惯比盲目运行代码更重要。
