# Week 2 / Day 6：完整 NumPy 多特征线性回归

昨天训练的模型只有一个输入特征：

```text
prediction = weight * x + bias
```

今天把它扩展为多个特征：

```text
prediction = w1*x1 + w2*x2 + ... + bias
```

使用矩阵表示后：

```text
predictions = X @ weights + bias
```

## 1. 多特征数据的 shape

假设有 8 个样本，每个样本有 2 个特征：

```text
X.shape = (8, 2)
```

模型为每个特征准备一个权重：

```text
weights.shape = (2,)
```

偏置仍是一个数：

```text
bias = scalar
```

矩阵计算：

```text
(8, 2) @ (2,) -> (8,)
```

因此：

```text
predictions.shape = (8,)
targets.shape = (8,)
```

## 2. 前向传播

```python
predictions = X @ weights + bias
```

这一次计算会为所有样本生成预测。

例如：

```text
X 中每一行是一个样本
weights 中每个元素对应一个特征
X @ weights 会让每一行与 weights 做点积
bias 再广播加到每个预测上
```

这就是线性层的 forward，也叫前向传播。

## 3. Loss

```python
errors = predictions - targets
loss = np.mean(errors ** 2)
```

shape：

```text
predictions: (N,)
targets:     (N,)
errors:      (N,)
loss:        scalar
```

`N` 表示样本数量。

## 4. 多个权重的梯度

单特征时：

```python
grad_weight = 2 * np.mean(errors * xs)
```

多特征时，每个特征都需要一个梯度。可以使用矩阵乘法一次完成：

```python
grad_weights = (2 / N) * (X.T @ errors)
```

shape：

```text
X.shape          = (N, D)
X.T.shape        = (D, N)
errors.shape     = (N,)
grad_weights     = (D,)

(D, N) @ (N,) -> (D,)
```

`D` 表示特征数量。

最终得到 `D` 个梯度，正好与 `D` 个权重一一对应。

## 5. 为什么要使用 X.T？

原始 `X` 的结构是：

```text
行：样本
列：特征
```

我们想分别统计每个特征对所有样本误差的影响，因此需要把特征放到前面：

```text
X.T 的行：特征
X.T 的列：样本
```

于是：

```text
第 1 个特征的所有值 · 所有 errors -> 第 1 个权重的梯度
第 2 个特征的所有值 · 所有 errors -> 第 2 个权重的梯度
```

矩阵乘法把这些点积一次完成。

## 6. Bias 的梯度

```python
grad_bias = 2 * np.mean(errors)
```

因为同一个 bias 会加到每个样本的预测上，所以只需要对所有误差求平均。

## 7. 参数更新

```python
weights -= learning_rate * grad_weights
bias -= learning_rate * grad_bias
```

shape 保持不变：

```text
weights.shape      = (D,)
grad_weights.shape = (D,)
```

每个权重使用自己对应的梯度进行更新。

## 8. 完整训练循环

```python
for epoch in range(epochs):
    predictions = X @ weights + bias
    errors = predictions - targets
    loss = np.mean(errors ** 2)

    grad_weights = (2 / len(X)) * (X.T @ errors)
    grad_bias = 2 * np.mean(errors)

    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias
```

这就是一个完整的多特征线性回归训练器。

## 9. 用 class 组织模型

今天会把模型写成类：

```python
model = NumpyLinearRegression(n_features=2)
model.fit(X, targets, learning_rate=0.05, epochs=1000)
predictions = model.predict(X)
```

模型需要保存：

```text
weights
bias
loss_history
```

模型需要提供：

```text
predict()：前向预测
compute_loss()：计算 MSE
compute_gradients()：计算梯度
fit()：执行训练循环
```

## 10. 与神经网络的关系

今天的模型已经包含神经网络训练的核心结构：

```text
参数初始化
forward
loss
gradient
parameter update
重复训练
```

神经网络只是把一个线性模型扩展成很多层，并在层之间加入激活函数。

## 11. 今日 shape 表

```text
X              (N, D)
weights        (D,)
bias           scalar
predictions    (N,)
targets        (N,)
errors         (N,)
X.T            (D, N)
grad_weights   (D,)
grad_bias      scalar
loss           scalar
```

## 12. 今日关键词

```text
feature matrix：特征矩阵
weight vector：权重向量
forward：前向传播
gradient vector：梯度向量
fit：训练模型
loss history：损失历史
```
