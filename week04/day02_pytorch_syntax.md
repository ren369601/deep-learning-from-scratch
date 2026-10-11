# Week 4 / Day 2：Autograd 用到的 PyTorch 写法

## 1. 创建需要梯度的 Tensor

```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
```

`requires_grad=True` 常用于可学习的权重和偏置；普通输入、标签默认不需要。

## 2. 前向计算与标量损失

```python
prediction = x * w + b
loss = ((prediction - target) ** 2).mean()
print(loss.shape)  # torch.Size([])，标量 Tensor
```

对多个误差取平均时，`.mean()` 会把它们变成一个标量。`loss.item()` 可取出 Python 数值用于显示，不要在 `.backward()` 前把 loss 换成这个数值。

## 3. 反向传播与读取梯度

```python
loss.backward()
print(w.grad)
print(b.grad)
```

`.backward()` 不返回梯度；它把结果累积到需要梯度的叶子 Tensor 的 `.grad` 中。

## 4. 清空梯度与重新前向

```python
w.grad.zero_()       # 已有梯度时，在原位置清零
prediction = x * w + b
loss = ((prediction - target) ** 2).mean()
loss.backward()
```

只有 `.grad` 已存在时才能调用 `.zero_()`；首次反向前它一般是 `None`。第二次反向要重新计算 `prediction` 和 `loss`，不要对同一张图直接重复调用 `.backward()`。

## 5. 作业需要的矩阵写法

```python
prediction = X @ W + b
d_prediction = 2 * (prediction - target) / prediction.numel()
dW_manual = X.T @ d_prediction
db_manual = d_prediction.sum(dim=0)

print(W.grad.shape, dW_manual.shape)
print(torch.allclose(W.grad, dW_manual))
```

`prediction.numel()` 是 Tensor 的元素总数；`torch.allclose()` 适合比较浮点结果，不要求它们逐位完全相同。
