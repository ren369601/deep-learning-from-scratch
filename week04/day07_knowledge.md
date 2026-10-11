# Week 4 / Day 7：PyTorch 基础周复习

今天不引入新模型。建议先读本页和语法速查，再不看 Day 5-6 的实现完成 `day07_week04_review.py`。预计用时 90-120 分钟。

## 1. 从 NumPy 到 PyTorch，变了什么

Week 3 手写了两层 MLP 的前向和反向传播。Week 4 仍是同一条训练链：

```text
数据 -> 模型前向 -> logits -> loss -> 反向传播 -> 参数更新
```

区别是 Tensor 可以记录求导关系，Autograd 自动计算参数梯度；`nn.Module` 管理层和参数；优化器根据梯度更新参数。自动化没有改变数学问题，输入、输出、loss 和数据切分仍需自己判断。

## 2. 六天知识连成一张图

| 天数 | 核心概念 | 今天要能回答 |
|---|---|---|
| Day 1 | Tensor、dtype、shape、广播 | 批量输入和线性层输出各是什么 shape？ |
| Day 2 | Autograd、计算图、梯度累积 | 为什么第二次 backward 前通常要清空梯度？ |
| Day 3 | nn.Module、forward、参数注册 | 为什么在 `__init__` 中把层赋给 `self`？ |
| Day 4 | SGD/Adam、学习率、更新顺序 | `zero_grad`、`backward`、`step` 各做什么？ |
| Day 5 | Linear -> ReLU -> Linear | 为什么最后输出 logits，不先做 Softmax？ |
| Day 6 | 训练/验证、loss/accuracy | 验证集为何不能用于参数更新？ |

以 XOR 分类为例：`X` 是 `(N, 2)`，隐藏层输出 `(N, H)`，模型 logits 为 `(N, 2)`，标签为长度 `N` 的整数向量。`CrossEntropyLoss` 直接接收 logits 与整数类别标签。

## 3. 一次完整训练与验证

训练阶段使用训练数据，按顺序切换训练模式、清梯度、前向、计算 loss、反向、更新。更新后重新前向计算指标，才能把日志与当前参数对应起来。

验证阶段切到评估模式，并在 `torch.no_grad()` 中计算 loss 与 accuracy。`eval()` 影响 Dropout、BatchNorm 等层的行为；`no_grad()` 不记录求导图。它们都**不等于**“锁住参数”；真正不更新参数，是因为验证过程没有调用优化器的 `step()`。

训练集和验证集分别回答不同的问题：模型是否拟合了用来更新参数的数据，以及它对同分布但未参与更新的数据表现如何。XOR 四个角附近的小样本验证结果不能推论到所有输入空间。

## 4. 先口答，再打开代码

1. 若输入 `(20, 2)`，模型为 `Linear(2, 16) -> ReLU -> Linear(16, 2)`，中间层、logits 和标签各是什么 shape？参数总数是多少？
2. `model.train()` 与 `model.eval()` 会不会自动执行或停止参数更新？
3. 为什么不能在 `CrossEntropyLoss` 前手动加 Softmax？
4. 连续两次 `backward()` 而不清梯度，`.grad` 会怎样？
5. 为什么在 `optimizer.step()` 后要重新计算 train/val 指标？
6. 若 train loss 下降而 val loss 上升，应该先检查什么？能否仅凭一次小验证集波动就断定过拟合？

把你的答案和运行记录写在 `review.md`。遇到卡点，先回看对应的 Day 1-6 文件，不必为了赶进度跳过。

## 5. 今天的完成标准

- 能从 Tensor 的 shape 讲到模型 logits、loss、梯度和参数更新。
- 不参考 Day 6，能写出训练与验证的基本结构。
- 能解释 `train/eval`、`no_grad`、`zero_grad` 的不同职责。
- 综合作业在固定数据上运行，并报告训练/验证结果与局限。

完成 Week 4 复习后，再进入 Week 5 的老师课程内容。
