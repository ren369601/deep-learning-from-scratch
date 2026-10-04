# Week 3 / Day 5：NumPy 两层 MLP 前向传播

MLP 是 Multi-Layer Perceptron 的缩写，中文常译为多层感知机。

今天实现的结构：

```text
输入 X
-> Linear 1
-> ReLU
-> Linear 2
-> logits
-> Softmax
-> probabilities
```

## 1. 什么是两层 MLP？

“两层”通常指有参数的两层：

```text
第一层 Linear：输入层到隐藏层
第二层 Linear：隐藏层到输出层
```

ReLU 和 Softmax 没有可训练的 weight 和 bias，所以通常不计入有参数层的数量。

公式：

```text
Z1 = X @ W1 + b1
A1 = ReLU(Z1)
Z2 = A1 @ W2 + b2
P  = Softmax(Z2)
```

其中 `Z2` 也叫 logits，`P` 是分类概率。

## 2. Shape 流程

定义：

```text
N：batch size
D：输入特征数量
H：隐藏层神经元数量
C：类别数量
```

所有 shape：

```text
X       (N, D)
W1      (D, H)
b1      (H,)
Z1      (N, H)
A1      (N, H)
W2      (H, C)
b2      (C,)
Z2      (N, C)
P       (N, C)
```

矩阵计算：

```text
(N, D) @ (D, H) + (H,) -> (N, H)
(N, H) @ (H, C) + (C,) -> (N, C)
```

## 3. 隐藏层是什么？

隐藏层是输入和最终输出之间的中间表示。

```text
原始输入 X
-> 组合成隐藏特征 A1
-> 使用隐藏特征进行分类
```

例如输入只有 2 个特征，但隐藏层可以有 8 个神经元：

```text
(N, 2) -> (N, 8) -> (N, 2)
```

隐藏层宽度 `H` 是我们选择的超参数。

## 4. 为什么 Linear 之间需要 ReLU？

如果没有 ReLU：

```text
Linear 1 -> Linear 2
```

两层可以合并成一个线性层，仍然无法表示复杂的非线性边界。

加入 ReLU 后：

```text
Linear 1 -> ReLU -> Linear 2
```

网络可以根据输入区域产生不同的响应，从而表示更复杂的函数。

## 5. 参数数量

第一层参数：

```text
W1：D * H
b1：H
```

第二层参数：

```text
W2：H * C
b2：C
```

总参数量：

```text
D*H + H + H*C + C
```

例如 `D=2, H=4, C=2`：

```text
2*4 + 4 + 4*2 + 2 = 22
```

## 6. 权重初始化

ReLU 网络常使用 He Initialization：

```python
scale = np.sqrt(2.0 / in_features)
weights = rng.normal(
    0.0,
    scale,
    size=(in_features, out_features),
)
```

它根据输入特征数量调整随机权重尺度，有助于让多层网络中的数值保持在合理范围。

今天先学会使用，后面会进一步理解初始化对训练的影响。

## 7. 为什么 Forward 要缓存数据？

明天计算 Backward 时需要：

```text
Linear 2 backward 需要 A1
ReLU backward 需要 Z1
Linear 1 backward 需要 X
```

因此 Forward 时保存：

```python
self.last_input = X
self.hidden_linear = Z1
self.hidden = A1
self.logits = Z2
```

这些缓存不是模型参数，而是当前 batch 前向传播产生的中间结果。

## 8. 随机初始化时预测没有意义

刚创建的模型还没有训练：

```text
weights 是随机值
probabilities 只是随机模型的输出
accuracy 可能碰巧较高或较低
```

训练的目标是通过 loss 和反向传播逐步调整参数。

今天只要求 Forward 正确，明天才让网络学习。

## 9. 完整 Forward

```python
def forward(self, X):
    self.hidden_linear = self.linear1.forward(X)
    self.hidden = self.relu.forward(self.hidden_linear)
    self.logits = self.linear2.forward(self.hidden)
    self.probabilities = softmax(self.logits)
    return self.probabilities
```

每一步都应该检查 shape。

## 10. 与 Transformer 的关系

Transformer 的 Feed Forward Network 也是类似结构：

```text
Linear -> Activation -> Linear
```

区别是 Transformer 会对每个 token 独立使用同一套 FFN 参数，并配合残差连接和 LayerNorm。

## 11. 今日关键词

```text
MLP：多层感知机
hidden layer：隐藏层
hidden size：隐藏层宽度
logits：分类原始分数
forward pass：前向传播
cache：缓存的中间值
He initialization：He 权重初始化
```
