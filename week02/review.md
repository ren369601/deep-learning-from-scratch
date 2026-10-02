# Week 2 复习：NumPy、矩阵与梯度下降

Week 2 的主线是把 Week 1 的 Python 循环改造成 NumPy 矩阵计算，并让模型通过梯度下降自动学习参数：

```text
array 与 shape
-> reshape、transpose、broadcasting
-> 点积与矩阵乘法
-> MSE loss
-> gradient descent
-> 多特征线性回归
```

## 1. Array、Shape、Ndim 与 Dtype

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
], dtype=np.float32)
```

```text
X.shape = (2, 3)
X.ndim = 2
X.dtype = float32
X.size = 6
```

在机器学习中，`(2, 3)` 通常表示：

```text
2 个样本，每个样本有 3 个特征。
```

## 2. Reshape 与 Transpose

```python
numbers = np.arange(12)
matrix = numbers.reshape(3, 4)
```

`reshape` 只能改变组织方式，不能改变元素总数。

```text
(12,) -> (3, 4)  可以，因为都是 12 个元素
(12,) -> (2, 7)  不可以，因为需要 14 个元素
```

二维转置：

```text
(3, 4) -> (4, 3)
```

三维交换轴：

```python
np.transpose(x, (0, 2, 1))
```

```text
(B, T, D) -> (B, D, T)
```

## 3. Broadcasting

```text
(3, 4) +    (4,) -> (3, 4)
(3, 4) +  (3, 1) -> (3, 4)
```

广播判断从最后一个维度开始：

```text
维度相等，或者其中一个是 1，或者其中一边缺少该维度。
```

广播很方便，但也可能让错误 shape 的程序继续运行，因此计算 loss 前要检查 shape。

## 4. 点积与矩阵乘法

```text
*：逐元素乘法
@：点积或矩阵乘法
```

矩阵乘法规则：

```text
(m, n) @ (n, p) -> (m, p)
```

中间两个维度必须相同，结果保留外侧维度。

批量线性预测：

```python
predictions = X @ weights + bias
```

```text
X:           (N, D)
weights:     (D,)
predictions: (N,)
```

## 5. MSE Loss

```python
errors = predictions - targets
squared_errors = errors ** 2
loss = squared_errors.mean()
```

一行写法：

```python
loss = np.mean((predictions - targets) ** 2)
```

```text
loss 越小，模型的整体预测越接近 targets。
loss = 0，表示所有预测都完全正确。
```

## 6. Axis

如果：

```text
errors.shape = (N, O)
```

那么：

```text
errors.mean()       -> scalar，所有元素的平均
errors.mean(axis=0) -> (O,)，每个输出分别平均
errors.mean(axis=1) -> (N,)，每个样本分别平均
```

记忆方式：被指定的轴会被聚合，并从结果 shape 中消失。

## 7. 梯度下降

单特征：

```python
grad_weight = 2 * np.mean(errors * xs)
grad_bias = 2 * np.mean(errors)
```

多特征：

```python
grad_weights = (2 / n_samples) * (X.T @ errors)
grad_bias = 2 * np.mean(errors)
```

参数更新：

```python
weights -= learning_rate * grad_weights
bias -= learning_rate * grad_bias
```

训练循环顺序：

```text
forward -> loss -> gradients -> update
```

## 8. 多特征线性回归 Shape 表

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

## 9. 与 Transformer 的连接

本周知识会直接出现在 Transformer 中：

```text
shape：追踪 batch、sequence、feature 维度
reshape / transpose：拆分和组合 attention heads
broadcasting：添加 bias、位置编码和 mask
matrix multiplication：计算 Q、K、V 和 Attention
loss：衡量预测质量
gradient descent：训练所有模型参数
```

## 10. 必须能回答的问题

1. `(3, 4)` 在一批特征数据中通常表示什么？
2. 为什么 12 个元素不能 reshape 成 `(2, 7)`？
3. `(2, 3) @ (3, 4)` 的结果 shape 是什么？
4. `a * b` 和 `a @ b` 有什么区别？
5. `mean(axis=0)` 会让哪个轴消失？
6. 为什么计算 loss 前要检查 predictions 和 targets 的 shape？
7. 梯度为负数时，使用 `parameter -= lr * gradient` 后参数如何变化？
8. 学习率太小或太大分别会怎样？
9. 为什么多特征梯度中需要 `X.T`？
10. 一个训练 epoch 的四个主要步骤是什么？

## 11. 常见错误

```text
reshape 前后元素总数不一致。
把逐元素乘法 * 当成矩阵乘法 @。
矩阵乘法的内侧维度不相同。
错误广播没有报错，却产生了错误结果。
axis=0 和 axis=1 混淆。
参数更新时使用加梯度，而不是减梯度。
只更新 weights，忘记更新 bias。
在更新参数后才计算另一个参数的梯度。
```

## 12. Week 2 完成标准

- [x] 能创建并解释一维、二维和三维 array。
- [x] 能追踪 reshape 和 transpose 前后的 shape。
- [x] 能判断两个 shape 是否可以广播。
- [x] 能判断矩阵乘法是否合法及结果 shape。
- [x] 能用 NumPy 计算 MSE 和不同 axis 的平均值。
- [x] 能解释梯度符号和学习率的作用。
- [x] 能独立写出多特征线性回归训练循环。
- [x] 能解释 `X.T @ errors` 的 shape 与含义。
