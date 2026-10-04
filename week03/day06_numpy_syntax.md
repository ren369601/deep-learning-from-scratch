# Week 3 / Day 6：反向传播用到的 NumPy 写法

## 1. 正确类别的行列索引

```python
rows = np.arange(len(targets))
correct_probabilities = probabilities[rows, targets]
```

`rows` 选择样本，`targets` 选择该样本的正确类别。

## 2. 用概率构造 logits 梯度

```python
grad_logits = probabilities.copy()
grad_logits[rows, targets] -= 1.0
grad_logits /= len(targets)
```

使用 `.copy()`：否则修改 `grad_logits` 时也会改动模型缓存的概率。

## 3. 沿 batch 维度求和

```python
grad_bias = grad_output.sum(axis=0)
```

如果 `grad_output.shape = (N, C)`，结果 shape 是 `(C,)`，每个输出神经元得到一个 bias 梯度。

## 4. 反向矩阵乘法

```python
grad_weights = inputs.T @ grad_output
grad_inputs = grad_output @ weights.T
```

```text
inputs:      (N, D)
weights:     (D, C)
grad_output: (N, C)

grad_weights:(D, C)
grad_inputs: (N, D)
```

## 5. ReLU 梯度掩码

```python
grad_before_relu = grad_after_relu * (relu_input > 0)
```

`relu_input > 0` 得到布尔数组；参与乘法时，`True` 相当于 1，`False` 相当于 0。

## 6. 检查训练指标

```python
predicted_classes = probabilities.argmax(axis=1)
accuracy = np.mean(predicted_classes == targets)
```

打印时可写：

```python
print(f"loss={loss:.4f}, accuracy={accuracy:.2f}")
```
