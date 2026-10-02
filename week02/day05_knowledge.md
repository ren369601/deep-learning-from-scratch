# Week 2 / Day 5：梯度下降直觉

之前寻找最佳参数的方法是：

```text
尝试很多组 weight 和 bias
-> 分别计算 loss
-> 选择 loss 最小的一组
```

这种枚举方法在参数很少时可以使用，但神经网络可能有数百万甚至更多参数，不可能尝试所有组合。

今天学习一种更聪明的方法：梯度下降。

## 1. 梯度下降要解决什么问题？

我们的线性模型是：

```text
prediction = weight * x + bias
```

损失函数是：

```text
MSE = mean((prediction - target)^2)
```

训练的目标是找到合适的 `weight` 和 `bias`，让 MSE 尽可能小。

梯度下降不会一次知道最佳答案，而是重复进行：

```text
预测
-> 计算 loss
-> 计算参数应该向哪个方向移动
-> 更新参数
-> 再次预测
```

## 2. 什么是梯度？

可以先把梯度理解为：

```text
参数稍微变化时，loss 会朝哪个方向变化，以及变化有多快。
```

对于一个参数：

```text
gradient > 0：参数增大会让 loss 增大，所以应该减小参数。
gradient < 0：参数增大会让 loss 减小，所以应该增大参数。
gradient 接近 0：当前方向已经接近较低的位置。
```

因此更新公式是：

```text
parameter = parameter - learning_rate * gradient
```

注意是减去梯度，因为我们要让 loss 下降。

## 3. 什么是 learning rate？

`learning_rate`，简称 `lr`，叫作学习率。它控制每次更新走多大一步。

```text
学习率太小：每次移动很少，训练很慢。
学习率合适：loss 稳定下降。
学习率太大：可能跨过最低点，来回震荡，甚至越来越差。
```

例如：

```python
learning_rate = 0.01
```

参数更新：

```python
weight = weight - learning_rate * grad_weight
bias = bias - learning_rate * grad_bias
```

也常简写为：

```python
weight -= learning_rate * grad_weight
bias -= learning_rate * grad_bias
```

## 4. 线性回归的梯度

设：

```text
predictions = weight * xs + bias
errors = predictions - targets
```

MSE 对 `weight` 的梯度：

```text
grad_weight = 2 * mean(errors * xs)
```

MSE 对 `bias` 的梯度：

```text
grad_bias = 2 * mean(errors)
```

NumPy 实现：

```python
errors = predictions - targets
grad_weight = 2 * np.mean(errors * xs)
grad_bias = 2 * np.mean(errors)
```

今天先理解这两个公式的作用，不要求独立推导。

## 5. 为什么 weight 的梯度要乘 xs？

模型是：

```text
prediction = weight * x + bias
```

`weight` 对预测的影响会随着 `x` 改变：

```text
x 大时，weight 改一点，prediction 变化较大。
x 为 0 时，weight 改变不会影响 prediction。
```

所以计算 `weight` 的梯度时需要乘 `xs`。

而 `bias` 无论面对什么 `x`，都会直接加到预测上，所以它的梯度不需要乘 `xs`。

## 6. 一次完整更新

```python
predictions = weight * xs + bias
loss = np.mean((predictions - targets) ** 2)

errors = predictions - targets
grad_weight = 2 * np.mean(errors * xs)
grad_bias = 2 * np.mean(errors)

weight -= learning_rate * grad_weight
bias -= learning_rate * grad_bias
```

这就是一次梯度下降步骤。

## 7. 什么是 epoch？

使用整份训练数据完成一次预测、计算梯度和参数更新，通常称为一个 epoch。

```python
for epoch in range(1000):
    predictions = weight * xs + bias
    ...
    weight -= learning_rate * grad_weight
    bias -= learning_rate * grad_bias
```

随着 epoch 增加，我们希望看到：

```text
loss 逐渐下降
weight 和 bias 逐渐接近真实参数
```

## 8. 参数必须同时更新

先使用当前 `weight` 和 `bias` 计算两个梯度：

```python
grad_weight, grad_bias = compute_gradients(...)
```

然后再更新两个参数：

```python
weight -= learning_rate * grad_weight
bias -= learning_rate * grad_bias
```

不要更新完 `weight` 后，用新 `weight` 重新计算 `bias` 的梯度。两个梯度应该对应同一个旧模型状态。

## 9. 一次训练的完整流程

```text
初始化参数
    weight = 0
    bias = 0

重复多个 epoch：
    1. forward：计算 predictions
    2. loss：计算 MSE
    3. gradient：计算 grad_weight 和 grad_bias
    4. update：更新 weight 和 bias

得到训练后的参数
```

以后 PyTorch 会把梯度计算自动完成，但整个训练顺序不会改变。

## 10. 与反向传播的关系

今天的梯度公式是我们直接写出来的：

```python
grad_weight = 2 * np.mean(errors * xs)
grad_bias = 2 * np.mean(errors)
```

神经网络更复杂，不适合手动为每个参数写公式。反向传播会使用链式法则，高效计算每个参数的梯度。

PyTorch 中则会看到：

```python
loss.backward()
optimizer.step()
```

它们做的事情，和今天的“计算梯度、更新参数”是同一条主线。

## 11. 今日关键词

```text
gradient：梯度
gradient descent：梯度下降
learning rate：学习率
epoch：训练轮次
forward：前向计算
update：参数更新
training：训练
```
