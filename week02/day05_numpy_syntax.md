# Week 2 / Day 5：训练循环所需语法

## 1. `range()` 与 epoch

```python
for epoch in range(1000):
    ...
```

`epoch` 会依次取 `0` 到 `999`。

如果希望从 1 开始显示：

```python
print(epoch + 1)
```

## 2. 参数更新简写

```python
weight = weight - learning_rate * grad_weight
```

可以简写为：

```python
weight -= learning_rate * grad_weight
```

`bias` 同理。

## 3. 函数返回多个值

```python
def compute_gradients(...):
    return grad_weight, grad_bias
```

接收时：

```python
grad_weight, grad_bias = compute_gradients(...)
```

这叫拆包，前面已经在最佳参数搜索中使用过类似写法。

## 4. 每隔一段时间打印

如果每个 epoch 都打印，输出会很多。可以使用 `%`：

```python
if epoch % 100 == 0:
    print(epoch, loss)
```

`%` 表示取余数。能够被 100 整除的 epoch 才会打印。

## 5. 格式化小数

```python
print(f"loss={loss:.6f}")
```

`.6f` 表示以小数形式显示，并保留 6 位小数。

## 6. 训练循环模板

```python
weight = 0.0
bias = 0.0

for epoch in range(epochs):
    predictions = weight * xs + bias
    loss = np.mean((predictions - targets) ** 2)

    errors = predictions - targets
    grad_weight = 2 * np.mean(errors * xs)
    grad_bias = 2 * np.mean(errors)

    weight -= learning_rate * grad_weight
    bias -= learning_rate * grad_bias
```

要能说出这四个阶段：

```text
预测 -> loss -> gradient -> update
```
