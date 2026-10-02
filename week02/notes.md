# Week 2 Notes

## 本周主题

```text
NumPy、shape、矩阵乘法、MSE、梯度下降和多特征线性回归
```

## Day 1：NumPy Array

- [x] 创建一维和二维 array。
- [x] 查看 `shape`、`ndim`、`dtype` 和 `size`。
- [x] 使用二维索引选择行、列和单个元素。
- [x] 理解 `(3, 4)` 可以表示 3 个样本、每个样本 4 个特征。

## Day 2：Shape 操作

- [x] 使用 `reshape()` 改变数据组织方式。
- [x] 使用 `.T` 和 `np.transpose()` 交换轴。
- [x] 区分按特征广播和按样本广播。
- [x] 完成 `(3, 4) -> (4, 3)` 的综合流程。

## Day 3：点积与矩阵乘法

- [x] 区分 `*` 与 `@`。
- [x] 使用三种方法计算点积。
- [x] 使用矩阵乘法完成批量预测。
- [x] 完成 `(4, 3) @ (3, 2) -> (4, 2)` 的多输出计算。

## Day 4：MSE Loss

- [x] 使用 NumPy 分步计算 errors、squared errors 和 MSE。
- [x] 比较两个模型的 loss。
- [x] 使用 `axis=0` 计算每个输出的 MSE。
- [x] 使用 `axis=1` 计算每个样本的 MSE。

## Day 5：梯度下降

- [x] 理解梯度符号与参数更新方向。
- [x] 比较不同学习率的收敛速度。
- [x] 独立写出训练循环。
- [x] 成功学到 `weight=3`、`bias=2`，loss 接近 0。

## Day 6：多特征线性回归

- [x] 实现 `NumpyLinearRegression` 类。
- [x] 使用 `X.T @ errors` 一次计算全部权重梯度。
- [x] 两特征模型学到 `[2, -3]` 和 bias `5`。
- [x] 三特征模型学到 `[1.5, -2, 0.5]` 和 bias `4`。

## 已掌握的完整流程

```text
数据 X
-> 矩阵前向计算
-> predictions
-> MSE loss
-> gradients
-> 参数更新
-> 重复训练
```

## 仍需持续强化

- 每次矩阵计算前先写 shape。
- 区分 reshape、transpose 和 broadcasting。
- 区分逐元素乘法和矩阵乘法。
- 理解 axis 表示聚合掉哪个维度。
- 牢记训练顺序：forward、loss、gradients、update。

## Day 7

- [x] 完成 `day07_week02_review.py`。
- [x] 完成 shape、广播和矩阵乘法综合练习。
- [x] 从头实现 forward、MSE、gradients 和 train。
- [x] 模型学到 weights `[2, -1.5, 0.5]` 和 bias `3`。

综合作业结果：

```text
reshape: (24,) -> (2, 3, 4)
transpose: (2, 3, 4) -> (2, 4, 3)
broadcast result: (2, 4, 3)

matrix multiplication:
(4, 3) @ (3, 2) + (2,) -> (4, 2)

training:
first loss = 21.59375
final loss ≈ 0
learned weights ≈ [2.0, -1.5, 0.5]
learned bias ≈ 3.0
```

## Week 2 总结

本周已经能够使用 NumPy 完成从数据表示到模型训练的完整过程，并能通过 shape 判断矩阵运算是否合理。

下一周将进入神经网络基础：神经元、线性层、激活函数、损失函数和反向传播。
