# Week 5 / Day 1：用 PyTorch 看一项学习任务

今天不学新的 PyTorch API，而是把 Week 4 的写法和“任务六要素”对应起来。示例与作业使用不同数据，不要把下面的片段直接粘成作业答案。

## 1. 数据和模型

```python
import torch
from torch import nn

X_train = torch.tensor([[-1.0], [0.0], [1.0]])
y_train = torch.tensor([[-1.0], [1.0], [3.0]])
model = nn.Linear(1, 1)
print(X_train.shape)  # (3, 1)
print(model(X_train).shape)  # (3, 1)
```

`X_train` 是样本特征，`y_train` 是监督学习的目标。`nn.Linear(1, 1)` 让模型学习一个输入特征到一个数值的关系。形状相同才能逐元素比较预测与目标。

## 2. 损失与优化

```python
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

model.train()
optimizer.zero_grad()
predictions = model(X_train)
loss = loss_fn(predictions, y_train)
loss.backward()
optimizer.step()
```

`MSELoss` 衡量平均平方误差；`backward()` 计算梯度；`step()` 按优化器规则更新参数。日志中若写 `loss.item()`，它是本次更新**之前**前向计算的 loss；要报告更新后的 loss，需再次前向。

## 3. 留出数据与评价

```python
X_val = torch.tensor([[0.5]])
y_val = torch.tensor([[2.0]])

model.eval()
with torch.no_grad():
    val_predictions = model(X_val)
    val_mae = (val_predictions - y_val).abs().mean().item()
```

MAE 是平均绝对误差，单位与目标值相同。验证集不参与 `backward()` 或 `optimizer.step()`。若用验证集反复调参，它就不再是完全独立的最终测试数据。

## 4. 数据与输出的对照表

| 代码对象 | 任务元素 | 检查点 |
|---|---|---|
| `X_train`, `y_train` | 训练数据 | shape、dtype、数据来源 |
| `model` | 模型 | 输入/输出维度 |
| `loss_fn` | 训练目标 | 数值越小代表什么 |
| `optimizer` | 参数更新 | 是否只用训练数据 |
| `X_val`, `y_val` | 留出数据 | 未参与参数更新 |
| `val_mae` | 评价 | 是否适合实际任务 |

在 `week05` 目录运行今日练习：`python day01_course_map.py`。先预测各项 shape，再看程序输出。
