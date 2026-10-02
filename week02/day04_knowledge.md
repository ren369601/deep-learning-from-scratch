# Week 2 / Day 4：用 NumPy 计算 MSE Loss

Week 1 中，我们已经学习过平均平方误差：

```text
MSE = 所有 (prediction - target)^2 的平均值
```

当时使用 Python 循环逐个计算。今天改用 NumPy 对整批数据一次完成。

## 1. Prediction 与 Target

假设模型对 4 个样本的预测是：

```python
predictions = np.array([2.5, 5.0, 8.5, 10.0])
```

真实答案是：

```python
targets = np.array([3.0, 5.0, 7.0, 9.0])
```

这两个 array 的位置一一对应：

```text
predictions[0] 对应 targets[0]
predictions[1] 对应 targets[1]
```

因此它们的 shape 必须相同。

## 2. 第一步：计算误差

```python
errors = predictions - targets
```

NumPy 会逐元素相减：

```text
[-0.5, 0.0, 1.5, 1.0]
```

负数表示预测偏小，正数表示预测偏大。

## 3. 第二步：误差平方

```python
squared_errors = errors ** 2
```

结果：

```text
[0.25, 0.0, 2.25, 1.0]
```

平方的作用：

```text
避免正负误差互相抵消
让较大的错误受到更明显的惩罚
结果一定不小于 0
```

## 4. 第三步：求平均值

```python
loss = squared_errors.mean()
```

计算：

```text
(0.25 + 0 + 2.25 + 1) / 4 = 0.875
```

所以：

```text
MSE = 0.875
```

可以写成一行：

```python
loss = np.mean((predictions - targets) ** 2)
```

## 5. Loss 为什么通常是一个数？

一个 batch 中有很多样本，也可能有多个输出，但训练时通常需要一个总体指标来判断当前模型表现。

因此：

```text
predictions：可能是一批数据
errors：每个位置各有一个误差
loss：把所有误差汇总成一个标量
```

这个标量越小，说明整批预测整体上越接近目标。

## 6. `sum()` 与 `mean()`

```python
values = np.array([1, 2, 3, 4])
```

```python
values.sum()   # 10
values.mean()  # 2.5
```

二者区别：

```text
sum：把所有值相加
mean：相加后除以元素数量
```

MSE 使用平均值，因此 batch 大小改变时，loss 通常仍保持在可比较的数值范围。

## 7. 什么是 axis？

对于二维数据：

```python
values = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

shape 是 `(2, 3)`。

对所有元素求平均：

```python
values.mean()
```

结果是一个数：`3.5`。

沿 `axis=0` 求平均：

```python
values.mean(axis=0)
```

结果：

```text
[2.5, 3.5, 4.5]
```

可以理解为压缩第 0 轴，对每一列跨样本求平均。

沿 `axis=1` 求平均：

```python
values.mean(axis=1)
```

结果：

```text
[2.0, 5.0]
```

可以理解为压缩第 1 轴，对每一行求平均。

## 8. 多输出模型中的 MSE

假设 3 个样本，每个样本有 2 个输出：

```text
predictions.shape = (3, 2)
targets.shape = (3, 2)
```

总体 MSE：

```python
overall_mse = ((predictions - targets) ** 2).mean()
```

每个输出各自的 MSE：

```python
per_output_mse = ((predictions - targets) ** 2).mean(axis=0)
```

结果 shape 是 `(2,)`，因为有两个输出。

每个样本各自的 MSE：

```python
per_sample_mse = ((predictions - targets) ** 2).mean(axis=1)
```

结果 shape 是 `(3,)`，因为有三个样本。

## 9. 小心意外广播

如果：

```text
predictions.shape = (3, 1)
targets.shape = (3,)
```

二者相减时可能广播成 `(3, 3)`，程序不会一定报错，但计算含义已经错了。

因此损失函数中最好检查：

```python
if predictions.shape != targets.shape:
    raise ValueError("predictions 和 targets 的 shape 必须相同")
```

这是一类很常见、也很隐蔽的深度学习错误。

## 10. 与模型训练的关系

完整训练过程将是：

```text
inputs
-> 模型预测 predictions
-> 与 targets 比较
-> 计算 loss
-> 根据 loss 调整参数
-> 再次预测
```

今天掌握了中间的 loss。下一天会开始学习如何根据 loss 调整参数。

## 11. 今日关键词

```text
prediction：预测值
target / label：真实值
error：误差
squared error：平方误差
MSE：平均平方误差
loss：损失
axis：轴
scalar：标量
```
