# Week 4 / Day 4：Optimizer、SGD 与 Adam

Day 2 的 `loss.backward()` 只计算梯度；Day 3 的 `nn.Module` 帮我们集中管理参数。今天把两者接起来，让参数真正更新。

## 1. 一轮训练的顺序

```python
optimizer.zero_grad()
prediction = model(X)
loss = loss_fn(prediction, target)
loss.backward()
optimizer.step()
```

`zero_grad()` 清掉上轮梯度；`backward()` 把本轮梯度写进参数的 `.grad`；`step()` 用优化器规则更新参数。`step()` 之后若想知道**新参数**对应的 loss，必须重新前向计算。PyTorch 官方训练教程也使用清梯度、反向、更新这三个步骤：[优化模型参数](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。

## 2. SGD：先理解最简单的更新

没有动量、权重衰减等选项时，SGD 就是你在 Week 2 手写的规则：

```text
parameter_new = parameter_old - learning_rate * gradient
```

例如一个 `nn.Linear(1, 1)` 初始 `weight=0`、`bias=0`，单个样本 `x=1`、`target=3`，平方误差 `loss=(prediction-target)^2=9`。

```text
dLoss/dweight = -6
dLoss/dbias   = -6
learning_rate = 0.1
更新后 weight = 0.6，bias = 0.6
```

下面代码可以复现这一步：

```python
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

`model.parameters()` 把模型里的可学习参数交给优化器。若后续增加层，优化器也需要拿到那些参数。

## 3. Adam：使用梯度历史

Adam 不只是把当前梯度乘学习率。它维护梯度的一阶、二阶移动统计，并据此调整更新幅度。今天只需会调用和比较，不用背推导：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

SGD 和 Adam 的同名 `lr` 不表示它们会走出一样大的更新。比较时应保持相同模型初值、数据、损失函数与训练轮数，并分别记录曲线；不要预设 Adam 在每个小问题上一定更好。

## 4. 为什么每轮都要清梯度

PyTorch 默认累积梯度。若忘了 `optimizer.zero_grad()`，第二轮的 `.grad` 会含有第一轮的梯度，实际更新就不再是你预期的“只根据当前 batch 更新”。

```python
for epoch in range(10):
    optimizer.zero_grad()
    prediction = model(X)
    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()
```

这里把整个数据集当一个 batch。以后 Week 5 学 `DataLoader` 后，每个 batch 都执行同样的顺序。PyTorch 可能把清零后的 `.grad` 设为 `None`，不必假定它一定是全零 Tensor。

## 5. 今天的易错点

1. `loss.backward()` 不更新参数；`optimizer.step()` 才更新。
2. 每轮先清梯度，再新建前向计算图；不要重复对同一个 `loss` 反向传播。
3. 比较更新前后 loss 时，更新后要再算一次 `model(X)`。
4. 记录 loss 时注明是“本轮更新前”还是“更新后”。
5. 不要一边用优化器 `step()`，一边手动 `weight -= lr * grad`，否则会重复更新。

## 6. 今日完成标准

- 能解释 `zero_grad()`、`backward()`、`step()` 分别做什么。
- 能手算并复现一次没有动量的 SGD 更新。
- 能写一个短训练循环并观察 loss 变化。
- 能在相同初始条件下比较 SGD 与 Adam，并用实验结果而不是印象下结论。

学习顺序：本页 -> `day04_pytorch_syntax.md` -> `day04_optimizer.py`。明天把今天的训练步骤用于两层 MLP。
