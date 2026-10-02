# Week 2 / Day 6：多特征训练语法

## 1. 根据特征数初始化权重

```python
n_features = 3
weights = np.zeros(n_features, dtype=np.float32)
```

结果：

```text
[0. 0. 0.]
shape = (3,)
```

## 2. 获取样本数量

```python
n_samples = X.shape[0]
```

如果 `X.shape = (8, 2)`，那么：

```text
n_samples = 8
```

## 3. 矩阵梯度

```python
grad_weights = (2 / n_samples) * (X.T @ errors)
```

计算前后：

```text
X.T:          (D, N)
errors:       (N,)
grad_weights: (D,)
```

## 4. 保存 loss 历史

```python
loss_history = []

for epoch in range(epochs):
    loss = ...
    loss_history.append(loss)
```

训练后可以检查：

```python
print(loss_history[0])
print(loss_history[-1])
```

如果训练正常，最后的 loss 应明显小于最初的 loss。

## 5. 输入 shape 检查

```python
if X.ndim != 2:
    raise ValueError("X 必须是二维数组")

if X.shape[1] != self.weights.shape[0]:
    raise ValueError("特征数量和权重数量不一致")
```

## 6. 参数副本

如果需要在返回时避免外部代码意外修改模型参数，可以使用：

```python
return self.weights.copy()
```

今天不是必须，但要知道 NumPy array 是可修改对象。

## 7. 完整使用方式

```python
model = NumpyLinearRegression(n_features=2)

history = model.fit(
    X,
    targets,
    learning_rate=0.05,
    epochs=1000,
)

predictions = model.predict(X)
loss = model.compute_loss(predictions, targets)
```

## 8. `@staticmethod`

今天的 `compute_loss()` 只需要 `predictions` 和 `targets`，不需要读取 `self.weights` 或 `self.bias`。

因此可以写成静态方法：

```python
@staticmethod
def compute_loss(predictions, targets):
    return float(np.mean((predictions - targets) ** 2))
```

调用方式仍然很自然：

```python
loss = model.compute_loss(predictions, targets)
```

先简单记住：

```text
普通实例方法需要 self。
静态方法不需要 self，也不直接使用对象内部的属性。
```
