# Week 4 / Day 5：用 PyTorch 搭建两层 MLP

Week 3 用 NumPy 手写了 `Linear -> ReLU -> Linear -> Softmax`。今天用 `nn.Module` 重新搭建前面的网络结构，并让 Autograd 计算梯度。完整训练与验证留到 Day 6。

## 1. 两层 MLP 的数据流

对一批输入 `X.shape=(N, D)`，隐藏层宽度为 `H`，类别数为 `C`：

```text
X                  (N, D)
  -> nn.Linear(D,H) (N, H)
  -> nn.ReLU()      (N, H)
  -> nn.Linear(H,C) (N, C) = logits
```

中间的 ReLU 引入非线性；没有它，两层线性变换可合并成一层，难以表达 XOR。今天的 `forward()` **返回 logits，不返回 Softmax 概率**。

## 2. 为什么不在模型末尾放 Softmax

分类训练使用 `nn.CrossEntropyLoss()`：

```python
logits = model(X)
loss = nn.CrossEntropyLoss()(logits, targets)
```

这个损失函数接受未归一化的 logits，并在内部处理与 Softmax/对数有关的计算。若先把模型输出转成概率再传入，会让输入语义不对。只在观察结果时另算概率：

```python
probabilities = torch.softmax(logits, dim=1)
predicted_classes = logits.argmax(dim=1)
```

`targets` 是每个样本的类别编号，通常是 `torch.long`、shape 为 `(N,)`；`logits` 的 shape 是 `(N, C)`。[PyTorch CrossEntropyLoss 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)明确要求以未归一化 logits 作为输入。

## 3. PyTorch 权重 shape 与 NumPy 的区别

`nn.Linear(D,H)` 内部的 `weight.shape` 是 `(H,D)`，因为计算等价于 `X @ weight.T + bias`。对 `D=2,H=8,C=2`：

```text
linear1.weight    (8, 2)
linear1.bias      (8,)
linear2.weight    (2, 8)
linear2.bias      (2,)
logits            (N, 2)
```

参数总数是 `8*2 + 8 + 2*8 + 2 = 42`。Week 3 手写 NumPy 权重时采用 `(输入, 输出)`；这里的 weight 方向恰好相反，但模型的输入输出 shape 没变。

## 4. 今天只验证前向与梯度

用 XOR 的四个样本检查模块是否连接正确：

```text
X       (4, 2)
targets (4,) = [0, 1, 1, 0]
logits  (4, 2)
loss    标量
```

调用 `loss.backward()` 后，四组参数的 `.grad` 应该都不是 `None`，且每组梯度的 shape 与对应参数一致。**随机初始化时预测不一定正确**；今天不以准确率作为通过标准。

`nn.Module` 自动登记层和参数，Autograd 自动记录运算，所以不需要像 Week 3 那样自己保存 `last_input` 或手写梯度公式。但你仍应能解释这些梯度在概念上来自哪里。

## 5. 今日完成标准

- 自己写出 `TwoLayerMLP(nn.Module)` 的 `__init__()` 与 `forward()`。
- 解释为什么 `forward()` 返回 logits，Cross Entropy 前不要手动 Softmax。
- 核对四组参数与梯度的 shape，算出参数总数 42。
- 在 XOR 上完成一次前向、损失和反向传播，不更新参数。

学习顺序：本页 -> `day05_pytorch_syntax.md` -> `day05_pytorch_mlp.py`。明天在此模型上写训练与验证循环。
