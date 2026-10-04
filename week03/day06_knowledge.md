# Week 3 / Day 6：训练两层 MLP

Day 5 的网络已经能计算：

```text
X -> Linear 1 -> ReLU -> Linear 2 -> logits -> Softmax -> probabilities
```

今天补上 Cross Entropy、反向传播和参数更新，让网络从数据中学习。先只处理一整批样本，不涉及 mini-batch。

## 1. 前向传播和损失

设 `N` 为样本数、`D` 为输入特征数、`H` 为隐藏神经元数、`C` 为类别数：

```text
X            (N, D)
W1           (D, H)
Z1           (N, H)
A1=ReLU(Z1)  (N, H)
W2           (H, C)
logits       (N, C)
P=Softmax    (N, C)
targets      (N,)     # 每个样本的正确类别下标
```

Cross Entropy 是所有样本的平均损失：

```text
loss = mean(-log(每个样本正确类别的概率))
```

## 2. 从输出开始反向传播

对于 Softmax 加平均 Cross Entropy，loss 对 logits 的梯度可以合并成：

```text
dZ2 = (P - one_hot(targets)) / N
```

`dZ2.shape = (N, C)`。`/ N` 来自“平均损失”，只在这里除一次。后面的矩阵乘法与求和不再重复除以 `N`。

例如正确类别是 1、预测概率是 `[0.2, 0.7, 0.1]`，那么该样本在除以 `N` 前的梯度为：

```text
[0.2, 0.7, 0.1] - [0, 1, 0] = [0.2, -0.3, 0.1]
```

## 3. 第二层 Linear 的梯度

第二层前向：

```text
Z2 = A1 @ W2 + b2
```

反向：

```text
dW2 = A1.T @ dZ2       # (H, N) @ (N, C) -> (H, C)
db2 = dZ2.sum(axis=0)  # (C,)
dA1 = dZ2 @ W2.T       # (N, C) @ (C, H) -> (N, H)
```

`dA1` 不是模型参数的梯度，而是继续传给前一层的梯度。

## 4. ReLU 的梯度

```text
dZ1 = dA1 * (Z1 > 0)
```

shape 仍为 `(N, H)`。前向时 `Z1 <= 0` 的位置，当前课程约定 ReLU 导数为 0，因此这些位置不传递梯度。

## 5. 第一层 Linear 的梯度

第一层前向：

```text
Z1 = X @ W1 + b1
```

反向：

```text
dW1 = X.T @ dZ1        # (D, N) @ (N, H) -> (D, H)
db1 = dZ1.sum(axis=0)  # (H,)
```

每个参数的梯度 shape 应与参数本身完全相同：

```text
dW1.shape == W1.shape
db1.shape == b1.shape
dW2.shape == W2.shape
db2.shape == b2.shape
```

## 6. 更新顺序

先用旧参数算完所有梯度，再更新：

```python
gradients = backward(model, targets)
model.linear1.weights -= learning_rate * gradients["W1"]
model.linear1.bias -= learning_rate * gradients["b1"]
model.linear2.weights -= learning_rate * gradients["W2"]
model.linear2.bias -= learning_rate * gradients["b2"]
```

特别注意：`dA1 = dZ2 @ W2.T` 必须使用更新前的 `W2`。因此不要算完 `dW2` 就立刻修改 `W2`。

## 7. 一轮训练的时间顺序

```text
1. model.forward(X)       使用当前参数得到 probabilities，并保存中间值
2. cross_entropy(...)     计算当前 loss
3. backward(...)          从输出层向输入层计算所有梯度
4. apply_gradients(...)   更新四组参数
```

日志中的 loss 如果在第 4 步之前记录，代表“本轮更新前”的 loss。训练结束后再运行一次 forward，才能得到最终参数对应的 loss。

## 8. 从 Day 5 的缓存取值

```text
model.linear1.last_input  -> X
model.relu.last_input     -> Z1
model.linear2.last_input  -> A1
model.probabilities       -> P
```

这些缓存是“最近一次 forward 的结果”。每次更新参数后，需要重新 forward 才能得到新参数的预测。

## 9. 今日完成标准

- 能解释为什么 `dZ2` 要除以 `N`。
- 能按第二层、ReLU、第一层的顺序写出梯度。
- 能核对四组梯度与参数的 shape。
- 能在更新前算齐所有梯度。
- 能观察训练前后 loss 与 accuracy 的变化。

下一周学习 PyTorch 时，`loss.backward()` 会替我们完成今天手写的梯度计算，但训练顺序相同。
