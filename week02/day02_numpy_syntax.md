# Week 2 / Day 2：NumPy Shape 操作语法

## 1. `reshape()`

```python
numbers = np.arange(12)

matrix_a = numbers.reshape(3, 4)
matrix_b = numbers.reshape(2, 6)
matrix_c = numbers.reshape(2, 2, 3)
```

对应 shape：

```text
(3, 4)
(2, 6)
(2, 2, 3)
```

也可以传入 tuple：

```python
matrix = numbers.reshape((3, 4))
```

## 2. 自动推断维度

```python
matrix = numbers.reshape(3, -1)
flat = matrix.reshape(-1)
```

```text
matrix.shape = (3, 4)
flat.shape = (12,)
```

## 3. 二维转置 `.T`

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

transposed = matrix.T
```

```text
matrix.shape = (2, 3)
transposed.shape = (3, 2)
```

## 4. `np.transpose()`

二维时，下面两种写法效果相同：

```python
matrix.T
np.transpose(matrix)
```

三维以上的数据可以指定轴的新顺序：

```python
x = np.zeros((2, 3, 4))
y = np.transpose(x, (0, 2, 1))
```

结果：

```text
x.shape = (2, 3, 4)
y.shape = (2, 4, 3)
```

这里保留第 0 轴，把第 1 轴和第 2 轴交换。今天只需认识，后面多头注意力会重点练习。

## 5. 广播一个标量

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

print(matrix + 10)
```

数字 `10` 会应用到矩阵的每个元素。

## 6. 广播一行

```python
feature_bias = np.array([10, 20, 30])
result = matrix + feature_bias
```

`feature_bias` 会应用到矩阵的每一行。

## 7. 广播一列

```python
row_bonus = np.array([
    [100],
    [200],
])

result = matrix + row_bonus
```

每一行会加上各自对应的 bonus。

## 8. 观察 shape，而不是猜

遇到 shape 计算时，先打印：

```python
print("matrix:", matrix.shape)
print("feature_bias:", feature_bias.shape)
print("result:", result.shape)
```

Transformer 中排查错误时，这个习惯非常有用。
