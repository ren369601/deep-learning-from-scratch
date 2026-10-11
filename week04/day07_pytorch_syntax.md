# Week 4 / Day 7：PyTorch 一页速查

这页用于复习写法，不是综合作业的完整答案。写代码前先说出每一行的目的。

## Tensor 与 shape

```python
import torch
from torch import nn

X = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
print(X.shape, X.dtype)  # torch.Size([2, 2]), torch.float32
```

`nn.Linear(in_features, out_features)` 接受最后一维为 `in_features` 的输入；其 `weight` 形状为 `(out_features, in_features)`。

## 模型、参数与输出

```python
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
logits = model(X)
print(logits.shape)
print(sum(p.numel() for p in model.parameters()))
```

分类输出保持为 logits。`nn.CrossEntropyLoss()` 内部完成与 Softmax 相应的数值稳定计算，标签需为整数类别，dtype 通常是 `torch.long`。

## 自动求导与优化器

```python
targets = torch.tensor([1, 1], dtype=torch.long)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

model.train()
optimizer.zero_grad()
loss = loss_fn(model(X), targets)
loss.backward()
optimizer.step()
```

`zero_grad()` 清空已有梯度，不清空参数；`backward()` 计算并累积梯度；`step()` 才更新参数。不要用 `loss.item()` 的 Python 数值调用 `backward()`。

## 只评估，不更新

```python
model.eval()
with torch.no_grad():
    logits = model(X)
    loss_value = loss_fn(logits, targets).item()
    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
```

`eval()` 与 `no_grad()` 作用不同；这段代码没有 `optimizer.step()`，所以参数不会更新。再次训练前记得切回 `model.train()`。

## 常见自检

```text
X:        (N, D)
logits:   (N, C)
targets:  (N,)，整数类别
loss:     标量
param.grad.shape == param.shape
```

打印和记录时可用 `.item()`；计算梯度时保留 Tensor。若训练/验证指标要对应更新后的模型，必须在 `optimizer.step()` 之后重新前向计算。
