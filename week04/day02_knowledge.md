# Week 4 / Day 2：Autograd 自动求导

昨天用 Tensor 做了矩阵计算；今天让 PyTorch 帮我们做 Week 3 手写的反向传播。先理解一个事实：`loss.backward()` **计算梯度**，不会自动修改参数。

## 1. 从手算导数出发

设 `x=2`、`w=3`、`b=1`、`target=9`：

```text
prediction = x*w + b = 7
loss = (prediction - target)^2 = 4
dLoss/dprediction = 2*(7-9) = -4
dLoss/dw = -4*x = -8
dLoss/db = -4
```

这和 Week 3 Day 4 的链式法则是同一件事。PyTorch 根据前向运算建立计算图，从 loss 往回计算梯度。

```python
import torch

x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

prediction = x * w + b
loss = (prediction - 9.0) ** 2
loss.backward()

print(w.grad)  # tensor(-8.)
print(b.grad)  # tensor(-4.)
```

只给想学习的参数设置 `requires_grad=True`。这里 `x` 是固定输入，所以不需要为它求梯度。

## 2. `requires_grad`、计算图、`.grad`

前向计算把操作串起来：`w, b -> prediction -> loss`。`prediction` 和 `loss` 是中间结果，带有 `grad_fn`；`w` 和 `b` 是需要梯度的叶子 Tensor。调用 `loss.backward()` 后，梯度存放在 `w.grad` 和 `b.grad`。

```python
print(w.requires_grad)       # True
print(prediction.grad_fn)    # 非 None，具体名称无需记忆
print(w.grad.shape)          # 与 w.shape 一致
```

通常让 `loss` 成为一个标量（零维 Tensor）再调用无参数的 `.backward()`。如果输出是一组数，先用 `.mean()` 或 `.sum()` 形成符合题意的标量损失。

## 3. 梯度会累积

每次 `.backward()` 都会把新梯度**加到**叶子 Tensor 的 `.grad`，不会自动覆盖。若要重新计算一轮，先清空旧梯度，再做新的前向传播：

```python
w.grad.zero_()
prediction = x * w + b
loss = (prediction - 9.0) ** 2
loss.backward()
```

同一个计算图通常只反向传播一次；再次计算时重新前向建立图。下周更常用优化器的 `zero_grad()` 来清空梯度。

## 4. 一批数据的线性层梯度

对于 `prediction = X @ W + b`，若 `loss = mean((prediction - target)^2)` 且所有元素一起取平均：

```text
dPrediction = 2*(prediction - target) / prediction.numel()
dW = X.T @ dPrediction
db = dPrediction.sum(dim=0)
```

`/ prediction.numel()` 来自平均损失。今天的作业会让你同时得到 `W.grad`、`b.grad` 和手算矩阵公式的结果，并比较它们是否接近。

## 5. 今日完成标准

- 能解释 `requires_grad=True` 与 `.backward()` 各做什么。
- 能从 `.grad` 读取参数梯度，知道它与参数有相同 shape。
- 能解释为什么第二次反向前要清空梯度并重新前向。
- 能让一批数据的 Autograd 梯度与手写公式一致。

学习顺序：先读本页，再查 `day02_pytorch_syntax.md`，最后完成 `day02_autograd.py`。今天只算梯度，不做参数更新。官方参考：[Autograd 入门](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)、[backward API](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)。
