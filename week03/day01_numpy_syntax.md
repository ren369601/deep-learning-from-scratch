# Week 3 / Day 1：神经元与线性层语法

## 1. 单个神经元

```python
inputs = np.array([1, 2, 3], dtype=np.float32)
weights = np.array([0.5, -1.0, 2.0], dtype=np.float32)
bias = 1.0

output = inputs @ weights + bias
```

```text
(3,) @ (3,) + scalar -> scalar
```

## 2. 一批样本经过多个神经元

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
], dtype=np.float32)

weights = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
], dtype=np.float32)

bias = np.array([10, -5], dtype=np.float32)
output = X @ weights + bias
```

```text
(2, 3) @ (3, 2) + (2,) -> (2, 2)
```

## 3. 随机数生成器

```python
rng = np.random.default_rng(seed=42)
```

生成随机权重：

```python
weights = rng.normal(
    loc=0.0,
    scale=0.1,
    size=(in_features, out_features),
)
```

```text
loc：均值
scale：标准差
size：输出 shape
```

指定相同的 `seed`，每次运行会生成相同的随机结果。

## 4. 转为 float32

```python
weights = weights.astype(np.float32)
```

深度学习通常使用 `float32`，它比 `float64` 占用更少内存。

## 5. 浮点数比较

浮点数计算可能有微小误差，不建议直接使用 `==`：

```python
np.isclose(0.1 + 0.2, 0.3)
np.allclose(array_a, array_b)
```

`np.isclose()` 比较单个值，`np.allclose()` 比较整个 array。

## 6. LinearLayer 结构

```python
class LinearLayer:
    def __init__(self, in_features, out_features, seed=42):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            0.0,
            0.1,
            size=(in_features, out_features),
        ).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

    def forward(self, X):
        return X @ self.weights + self.bias
```

使用：

```python
layer = LinearLayer(in_features=3, out_features=2)
output = layer.forward(X)
```
