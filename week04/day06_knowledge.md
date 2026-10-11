# Week 4 / Day 6：训练与验证两层 MLP

Day 5 的 `TwoLayerMLP` 已能前向计算 logits 和反向求梯度。今天把 Day 4 的优化步骤放进循环，并单独检查模型在未参与更新的数据上的表现。

## 1. 训练集与验证集各做什么

- **训练集**：计算 loss、反向传播，用梯度更新参数。
- **验证集**：只评估当前模型；不调用 `backward()` 或 `optimizer.step()`。

原始 XOR 只有 4 个点。若从中拿走一个作验证，训练集就缺少一种模式，结果难解释。因此今天围绕四个 XOR 位置分别生成带小扰动的样本，每个位置用不同样本组成训练集与验证集。它仍只是一个很小的同分布玩具实验，不能据此声称模型具备真实世界泛化能力。

## 2. 训练循环

```python
model.train()
optimizer.zero_grad()
logits = model(X_train)
loss = loss_fn(logits, y_train)
loss.backward()
optimizer.step()
```

`model.train()` 把模型切回训练模式。当前 MLP 只有 Linear 和 ReLU，训练/评估模式不会改变它们的计算；但未来加入 Dropout 等层时会很重要。每轮先清旧梯度，再前向、求损失、反向、更新。

## 3. 验证循环

```python
model.eval()
with torch.no_grad():
    logits = model(X_val)
    val_loss = loss_fn(logits, y_val)
    val_acc = (logits.argmax(dim=1) == y_val).float().mean()
```

`model.eval()` 控制层的评估行为；`torch.no_grad()` 停止建立求导图。**它们不是同一个开关**，通常一起使用。验证时不更新参数。[PyTorch Autograd 说明](https://docs.pytorch.org/docs/stable/notes/autograd)也特别区分这两者。

## 4. Loss 与 Accuracy 的分工

`nn.CrossEntropyLoss()` 接收 `(N,C)` 的 logits 和 `(N,)` 的整数标签。Loss 能反映正确类别概率的变化；Accuracy 只统计预测类别是否正确。观察训练与验证时应同时记二者：

```text
train loss 下降、val loss 也下降：当前较健康
train loss 下降、val loss 上升：可能过拟合
两者都不下降：检查学习率、模型、数据和标签
```

这些是排查线索，不是绝对判决。尤其只有几十个验证样本时，Accuracy 会跳得很粗。今天不要根据验证集反复挑选参数再声称它是完全独立的最终测试结果。

## 5. 指标对应哪一组参数

若在 `optimizer.step()` 之前记录 loss，它对应**更新前**参数；更新后要重新 `model(X)` 才能报告更新后的 loss。今天的作业请在每个日志点对训练集与验证集都重新评估，让两个指标对应同一时刻的模型。

## 6. 今日完成标准

- 会计算从 logits 到分类 Accuracy。
- 能写一次正确的训练更新，并在更新后重新评估。
- 能用 `model.eval()` 与 `torch.no_grad()` 验证，且验证时参数不变。
- 能记录 train/val loss 与 accuracy，解释它们的差异和小样本局限。

学习顺序：本页 -> `day06_pytorch_syntax.md` -> `day06_train_validate.py`。明天复习本周标准训练模板。完整流程可对照 [PyTorch 训练与验证教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。
