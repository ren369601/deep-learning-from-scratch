# Week 3 / Day 4：反向传播与链式法则直觉

前几天已经完成神经网络的前向传播：

```text
输入 X
-> Linear
-> ReLU
-> Linear
-> Softmax
-> Loss
```

训练模型还需要解决一个关键问题：

```text
loss 已经算出来了，应该怎样调整每一个 weight 和 bias？
```

反向传播就是用来高效计算每个参数梯度的方法。

## 1. Forward 与 Backward

前向传播：

```text
输入 -> 中间结果 -> 预测 -> loss
```

反向传播：

```text
loss -> 预测的梯度 -> 中间结果的梯度 -> 参数的梯度
```

前向传播计算“模型输出什么”，反向传播计算“每个参数对 loss 负有多少责任”。

## 2. 什么是导数？

导数可以先理解为：

```text
一个变量稍微变化时，另一个变量会变化多少。
```

例如：

```text
y = 3x
```

当 `x` 增加 `0.1` 时，`y` 大约增加 `0.3`，所以：

```text
dy/dx = 3
```

常见导数：

```text
y = x       -> dy/dx = 1
y = x^2     -> dy/dx = 2x
y = ax + b  -> dy/dx = a
```

## 3. 什么是链式法则？

假设：

```text
z = w * x + b
loss = (z - target)^2
```

`w` 不直接出现在 loss 的最外层，而是先影响 `z`，再由 `z` 影响 loss：

```text
w -> z -> loss
```

链式法则：

```text
dLoss/dw = dLoss/dz * dz/dw
```

也就是把路径上每一小段的局部梯度乘起来。

## 4. 手算一个线性模型

给定：

```text
x = 2
w = 3
b = 1
target = 10
```

### Forward

```text
z = w*x + b
  = 3*2 + 1
  = 7

loss = (z - target)^2
     = (7 - 10)^2
     = 9
```

### Backward

先算 loss 对 `z` 的梯度：

```text
dLoss/dz = 2 * (z - target)
          = 2 * (7 - 10)
          = -6
```

`z` 对参数的局部梯度：

```text
dz/dw = x = 2
dz/db = 1
```

使用链式法则：

```text
dLoss/dw = dLoss/dz * dz/dw
          = -6 * 2
          = -12

dLoss/db = dLoss/dz * dz/db
          = -6 * 1
          = -6
```

所以：

```text
grad_w = -12
grad_b = -6
```

## 5. 用梯度更新参数

```text
parameter = parameter - learning_rate * gradient
```

如果学习率为 `0.1`：

```text
w_new = 3 - 0.1*(-12) = 4.2
b_new = 1 - 0.1*(-6)  = 1.6
```

因为梯度为负，减去负数后参数会增大。

## 6. 计算图

可以把计算画成一条图：

```text
x ----┐
      × -> wx ----┐
w ----┘           + -> z -> 减 target -> error -> 平方 -> loss
b ----------------┘
```

每个节点只需要知道两件事：

```text
Forward：怎样从输入计算输出。
Backward：上游梯度到来时，怎样传给自己的输入。
```

复杂神经网络就是更大的计算图。

## 7. ReLU 的梯度

```text
ReLU(z) = max(0, z)
```

它的导数：

```text
z > 0  -> 1
z <= 0 -> 0
```

因此反向传播时：

```python
grad_z = grad_output * (z > 0)
```

如果前向时某个值是负数，ReLU 把它变成 0；反向时这条路径的梯度也会被截断为 0。

在 `z = 0` 处数学导数没有唯一值，实际实现通常约定为 0。

## 8. 两层网络的反向传播

最小两层网络：

```text
z1 = w1*x + b1
a1 = ReLU(z1)
prediction = w2*a1 + b2
loss = (prediction - target)^2
```

Forward 顺序：

```text
x -> z1 -> a1 -> prediction -> loss
```

Backward 顺序完全相反：

```text
loss -> prediction -> a1 -> z1 -> w1、b1
```

梯度过程：

```text
dLoss/dPrediction = 2 * (prediction - target)

dLoss/dw2 = dLoss/dPrediction * a1
dLoss/db2 = dLoss/dPrediction
dLoss/da1 = dLoss/dPrediction * w2

dLoss/dz1 = dLoss/da1 * ReLU'(z1)
dLoss/dw1 = dLoss/dz1 * x
dLoss/db1 = dLoss/dz1
```

这就是链式法则沿计算图逐步向后传递。

## 9. 为什么 Forward 时要保存中间值？

Backward 会用到 Forward 的结果：

```text
计算 w2 的梯度需要 a1
计算 ReLU 梯度需要 z1
计算 w1 的梯度需要 x
```

因此神经网络层通常会在 Forward 时缓存必要的输入和中间结果，供 Backward 使用。

## 10. 数值梯度检查

如果不确定手写梯度是否正确，可以让参数稍微向两边移动：

```text
numeric_gradient =
    (loss(w + epsilon) - loss(w - epsilon)) / (2 * epsilon)
```

如果数值梯度与公式计算的解析梯度非常接近，说明梯度实现大概率正确。

```text
解析梯度：通过公式和反向传播计算
数值梯度：通过微小扰动参数估算
```

数值梯度速度慢，主要用于检查，不用于真正训练大型模型。

## 11. Backpropagation 的核心

请记住这五句话：

```text
1. Forward 从输入计算到 loss。
2. Backward 从 loss 沿原路径反向计算。
3. 链式法则把路径上的局部梯度相乘。
4. 每个参数最终得到一个与自己 shape 相同的梯度。
5. 优化器使用梯度更新参数。
```

## 12. 与 PyTorch 的关系

以后 PyTorch 会自动构建计算图：

```python
loss.backward()
```

这行代码会自动执行今天手算的反向传播。

```python
optimizer.step()
```

这行代码会根据梯度更新参数。

理解今天的过程后，这两行代码就不再像黑盒。

## 13. 今日关键词

```text
forward propagation：前向传播
backpropagation：反向传播
derivative：导数
gradient：梯度
chain rule：链式法则
computational graph：计算图
local gradient：局部梯度
numerical gradient：数值梯度
analytical gradient：解析梯度
```
