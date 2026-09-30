# Week 1 / Day 1 Notes

## 今天学了什么？

- 学习了最简单的线性模型：`y = w * x + b`。
- 理解了 `x`、`w`、`b`、`y` 分别表示什么。
- 学习了如何用 Python 函数表示一个模型。
- 学习了如何用 `list` 和 `for` 循环批量计算多个输入的输出。
- 学习了如何用 `class` 把模型参数 `w`、`b` 和预测方法 `predict()` 放在一起。

## 哪里不懂？

- 暂时没有明显卡住的地方。
- 后面还需要继续熟悉 `class`、`self`、`__init__` 这些 Python 语法。

## 线性模型 y = wx + b 是什么意思？

- `x` 是输入数据。
- `w` 是权重，决定输入 `x` 对输出的影响有多大。
- `b` 是偏置，用来让整体结果上移或下移。
- `y` 是模型输出，也就是预测结果。
- 模型学习的过程，本质上就是找到合适的 `w` 和 `b`，让预测结果更接近真实答案。

## 今日练习结果

```text
single value:
3 7
list values:
1 3
2 5
3 7
4 9
5 11

--- practice ---
1 5
2 8
3 11
4 14
5 17
6 20

--- homework_function ---
2 7
4 15
6 23
8 31
10 39

--- LinearModel class ---
7
```

## 今日作业

- [x] 完成 `practice()`
- [x] 完成 `homework_function()`
- [x] 完成 `LinearModel`
- [x] 能解释 `w`、`x`、`b`、`y` 分别是什么

## 今日小结

今天完成了 Week 1 / Day 1 的基础练习。

目前已经能够用 Python 写出最简单的线性模型，并用函数和类两种方式进行预测。

下一步可以继续学习更多 Python 基础和 NumPy，为之后理解矩阵计算、Linear Regression 和神经网络打基础。

---

# Week 1 / Day 2 Notes

## 今天学了什么？

- 学习了多输入线性模型：`y = w1 * x1 + w2 * x2 + ... + b`。
- 理解了 `feature`（特征）：模型用来做预测的输入信息。
- 理解了每个特征都要有一个位置对应的权重。
- 初步认识了点积：对应元素相乘后求和。
- 学习了 `zip()`，可以同时遍历 `features` 和 `weights`。
- 学习了 `+=`，可以持续累加计算结果。
- 使用 `class` 封装了多输入线性模型。

## 今天最重要的理解

- `features = [x1, x2, x3]` 表示一个样本的多个输入特征。
- `weights = [w1, w2, w3]` 表示每个特征各自对应的权重。
- `features` 和 `weights` 的长度必须相同，否则会有特征没有对应权重。
- 权重为正，说明特征增加时预测值倾向于增加；权重为负，说明特征增加时预测值倾向于减少。

## 今日练习结果

### 学生成绩预测

```text
features = [10, 8, 2]
weights = [5, 3, -4]
bias = 40

预测结果 = 10 * 5 + 8 * 3 + 2 * (-4) + 40 = 106
```

### 模型类作业

```text
weights = [2, 3]
bias = 1

[1, 2] -> 9
[3, 4] -> 19
```

## 今日作业

- [x] 完成 `practice_student_score()`
- [x] 完成 `MultiFeatureLinearModel`
- [x] 完成 `homework_model_class()`
- [x] 能解释特征、权重、偏置和点积的基本含义

## 今日小结

现在已经能用纯 Python 表示一个有多个特征的线性模型，并能用 `zip()` 和循环完成“特征乘权重再求和”的计算。

下一步会学习如何处理多个样本，让模型一次对一批数据做预测。这会为后面的矩阵和 NumPy 做准备。

---

# Week 1 / Day 3 Notes

## 今天学了什么？

- 学习了用嵌套 list 表示多个样本。
- 理解了一个样本、一批样本和一批预测结果之间的对应关系。
- 学习了 `append()` 和 `enumerate()`。
- 使用同一组 `weights` 和 `bias` 完成了批量预测。

## 今日练习结果

```text
学生成绩预测：[65, 68, 39]
自定义批量预测：[69, 47, 86]
```

自定义作业中组合使用了 `enumerate()` 和 `zip()`，能够同时取得编号、样本和对应预测结果。

## 今日作业

- [x] 完成 `practice_batch_prediction()`
- [x] 完成 `homework_custom_batch()`
- [x] 能解释 sample、batch 和 predictions 的关系

---

# Week 1 / Day 4 Notes

## 今天学了什么？

- 使用 `class` 把模型参数和预测方法组织在一起。
- 使用 `self.weights` 和 `self.bias` 保存模型自己的参数。
- 实现了单样本预测和批量预测。
- 使用 `set_parameters()` 手动更新模型参数。
- 理解了相同输入在不同模型参数下会得到不同预测。

## 今日练习结果

```text
单样本预测：65
批量预测：[65, 68, 39]
更新参数前：9
更新参数后：6
model_a 预测：10
model_b 预测：7
```

## 今日作业

- [x] 完成 `LinearModel`
- [x] 完成 `predict()` 和 `predict_batch()`
- [x] 完成 `set_parameters()`
- [x] 创建并比较两个独立模型

---

# Week 1 / Day 5 Notes

## 今天学了什么？

- 理解了输入、真实标签和模型预测之间的关系。
- 学习了平方误差和平均平方误差 MSE。
- 理解了 loss 越小，模型在当前数据上的预测通常越好。
- 使用双层循环枚举不同的 `weight` 和 `bias`。
- 使用 `best_loss` 保存目前找到的最佳结果。

## 今日练习结果

```text
weight=3, bias=0 -> MSE=3.5
weight=2, bias=2 -> MSE=1.0

最优 weight=2
最优 bias=1
最优 MSE=0.0
```

## 今日作业

- [x] 比较两组模型参数的 MSE
- [x] 使用双层循环遍历候选参数
- [x] 找出损失最小的线性模型

---

# Week 1 / Day 6 Notes

## 综合复习结果

独立完成了一个纯 Python 线性回归综合程序，包括：

- 用类保存 `weight` 和 `bias`。
- 单样本预测与批量预测。
- 计算整个数据集的 MSE。
- 枚举并比较候选参数。
- 使用最佳参数创建最终模型。

程序最终结果：

```text
预测值：[2, 5, 8, 11, 14]
best weight = 3
best bias = 2
best MSE = 0.0
```

## Week 1 总结

本周已经完成从 Python 基础到线性回归雏形的学习，能够理解并写出：

```text
数据 -> 模型 -> 预测 -> 损失 -> 参数比较 -> 最佳模型
```

下一周将进入 NumPy、数组 shape、矩阵计算和梯度下降，为神经网络与 Transformer 中的大规模数值计算打基础。
