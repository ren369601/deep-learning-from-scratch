# Week 3 / Day 5：组合多个类

## 1. 一个类持有其他层对象

```python
class TwoLayerMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.linear1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.linear2 = Linear(hidden_size, output_size)
```

这叫组合：MLP 对象内部保存多个层对象。

## 2. 逐层调用 Forward

```python
hidden_linear = self.linear1.forward(X)
hidden = self.relu.forward(hidden_linear)
logits = self.linear2.forward(hidden)
probabilities = softmax(logits)
```

每一层的输出会成为下一层的输入。

## 3. 使用 `None` 表示尚无缓存

```python
self.last_input = None
```

Forward 后再保存实际数据：

```python
self.last_input = X
```

这能清楚表示该层是否已经完成过前向传播。

## 4. He Initialization

```python
scale = np.sqrt(2.0 / in_features)
self.weights = rng.normal(
    0.0,
    scale,
    size=(in_features, out_features),
).astype(np.float32)
```

## 5. 计算参数数量

```python
parameter_count = (
    model.linear1.weights.size
    + model.linear1.bias.size
    + model.linear2.weights.size
    + model.linear2.bias.size
)
```

NumPy array 的 `.size` 是元素总数。

## 6. 检查概率

```python
row_sums = probabilities.sum(axis=1)
valid = np.allclose(row_sums, np.ones_like(row_sums))
```

多分类 Softmax 后，每个样本的概率和应接近 1。
