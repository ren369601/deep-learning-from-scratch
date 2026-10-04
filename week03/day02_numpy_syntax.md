# Week 3 / Day 2：激活函数所需 NumPy 语法

## 1. `np.maximum()`

```python
x = np.array([-2, -1, 0, 1, 2])
result = np.maximum(0, x)
```

它会逐元素选择较大的值，是 ReLU 的直接实现。

## 2. `np.exp()`

```python
x = np.array([0, 1, 2])
result = np.exp(x)
```

`np.exp(x)` 表示逐元素计算 `e^x`，用于 Sigmoid 和 Softmax。

## 3. `np.clip()`

```python
safe_x = np.clip(x, -500, 500)
```

小于 `-500` 的值变成 `-500`，大于 `500` 的值变成 `500`。这可以降低指数运算溢出的风险。

## 4. `np.tanh()`

```python
result = np.tanh(x)
```

NumPy 已提供稳定的 Tanh 实现。

## 5. `keepdims=True`

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

row_max = matrix.max(axis=-1, keepdims=True)
```

```text
matrix.shape = (2, 3)
row_max.shape = (2, 1)
```

保留长度为 1 的维度后，可以方便地广播回原矩阵。

## 6. 稳定 Softmax

```python
shifted = logits - np.max(logits, 
axis=-1, keepdims=True)
exp_values = np.exp(shifted)
probabilities = exp_values / np.sum(
    exp_values,
    axis=-1,
    keepdims=True,
)
```

检查每个样本的概率和：

```python
print(probabilities.sum(axis=-1))
```

结果应该接近全是 `1`。

## 7. `np.allclose()`

```python
row_sums = probabilities.sum(axis=-1)
is_valid = np.allclose(row_sums, np.ones_like(row_sums))
```

用于判断所有行的概率和是否都接近 1。
