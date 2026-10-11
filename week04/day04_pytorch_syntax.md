# Week 4 / Day 4：优化器用到的 PyTorch 写法

## 1. 损失函数和优化器

```python
import torch
from torch import nn

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
sgd = torch.optim.SGD(model.parameters(), lr=0.05)
adam = torch.optim.Adam(model.parameters(), lr=0.05)
```

实际训练一个模型时只选择一个优化器。比较 SGD 与 Adam 时，为它们分别创建**独立模型和优化器**。

## 2. 单次更新

```python
optimizer.zero_grad()
prediction = model(X)
loss = loss_fn(prediction, target)
loss.backward()
print(model.weight.grad, model.bias.grad)
optimizer.step()
```

`optimizer.zero_grad()` 可以把旧 `.grad` 置为 `None`；`backward()` 之后再读取本轮梯度。

## 3. 查看更新前后参数

```python
weight_before = model.weight.detach().clone()
optimizer.step()
weight_after = model.weight.detach().clone()
print(weight_before, weight_after)
```

此片段只示意保存快照，应放在**已经 backward、尚未 step** 的位置。用 `detach().clone()` 留下独立副本，不把打印和比较纳入计算图。

## 4. 短训练循环

```python
losses = []
for epoch in range(20):
    optimizer.zero_grad()
    prediction = model(X)
    loss = loss_fn(prediction, target)
    losses.append(loss.item())  # 此值是本轮更新前的 loss
    loss.backward()
    optimizer.step()

with torch.no_grad():
    final_loss = loss_fn(model(X), target).item()
```

`torch.no_grad()` 表示这里只评估结果，不需要建立求导图。以后验证集评估也会用到类似写法。

## 5. 公平比较两个优化器

```python
torch.manual_seed(42)
model_sgd = nn.Linear(1, 1)
model_adam = nn.Linear(1, 1)
model_adam.load_state_dict(model_sgd.state_dict())

optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.05)
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.05)
```

`load_state_dict()` 让两个模型从相同参数出发；两个优化器必须各自绑定对应模型。
