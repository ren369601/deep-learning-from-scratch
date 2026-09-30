# Week 1 / Day 3：批量样本与批量预测

Day 2 中，一个样本有多个特征：

```python
features = [80, 2, 10]
```

它表示一套房子的：

```text
面积、房间数、房龄
```

但训练模型时，我们不会只给模型一套房子的数据，而会给很多样本：

```python
samples = [
    [60, 1, 20],
    [80, 2, 10],
    [100, 3, 5],
]
```

这就是今天的重点：**一个样本是一行特征，多个样本组成一个 batch（批次）。**

## 1. 两层 list 表示什么？

```python
samples = [
    [60, 1, 20],
    [80, 2, 10],
    [100, 3, 5],
]
```

可以按层级读：

```text
samples
├── 第 1 个样本：[60, 1, 20]
├── 第 2 个样本：[80, 2, 10]
└── 第 3 个样本：[100, 3, 5]
```

其中：

```text
samples[0] = [60, 1, 20]
samples[1] = [80, 2, 10]
samples[2] = [100, 3, 5]
```

每个内部 list 都有 3 个特征，因此可以使用同一组长度为 3 的权重：

```python
weights = [0.8, 15, -1.2]
```

## 2. 什么是 batch？

`batch` 就是一小批同时处理的数据样本。

今天：

```text
batch = Python 里的多个样本组成的 list
```

以后使用 PyTorch 时，常见的形状会是：

```text
[batch_size, feature_size]
```

例如今天的三个房屋样本，每个样本有三个特征：

```text
[3, 3]
```

第一个 `3` 表示 3 个样本，第二个 `3` 表示每个样本有 3 个特征。

今天先建立这个直觉，不必急着记住 shape。

## 3. 如何批量预测？

我们已经会预测一个样本：

```python
prediction = multi_linear(features, weights, bias)
```

现在只要遍历所有样本即可：

```python
predictions = []

for features in samples:
    prediction = multi_linear(features, weights, bias)
    predictions.append(prediction)
```

循环结束后：

```python
predictions = [59.0, 102.0, 139.0]
```

这里的 `predictions` 与 `samples` 一一对应：

```text
samples[0] 的预测结果是 predictions[0]
samples[1] 的预测结果是 predictions[1]
samples[2] 的预测结果是 predictions[2]
```

## 4. 两层循环的关系

今天的批量预测，其实有两层计算：

```text
外层：逐个取出样本。
内层：在 multi_linear() 内，把该样本的每个特征和权重相乘后累加。
```

可以这样看：

```text
多个样本
  -> 取一个样本
      -> 计算这个样本的多个特征
  -> 取下一个样本
      -> 计算这个样本的多个特征
```

后面 NumPy 和 PyTorch 会把这类批量计算变得更快，但底层含义仍然是一样的。

## 5. 今天要形成的直觉

```text
一个样本：一组特征，例如 [80, 2, 10]。
一批样本：很多组特征，例如 [[60, 1, 20], [80, 2, 10]]。
批量预测：对每个样本使用同一组 weights 和 bias，得到一组 predictions。
```

## 6. 今日关键词

```text
sample：样本
batch：一批样本
batch size：一批中样本的数量
prediction：预测结果
predictions：一批预测结果
nested list：嵌套 list
```
