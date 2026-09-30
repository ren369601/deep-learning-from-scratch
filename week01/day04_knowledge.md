# Week 1 / Day 4：用 class 组织一个完整线性模型

前几天我们一直在写函数：

```python
prediction = multi_linear(features, weights, bias)
```

这没有问题，但每次预测都要重复传入 `weights` 和 `bias`。

今天我们把它改成一个模型对象：

```python
model = LinearModel(weights=[5, 3, -4], bias=40)
prediction = model.predict([2, 5, 0])
```

这样写的核心想法是：

```text
模型对象自己保存参数。
需要预测时，只把新的特征交给模型。
```

## 1. 为什么需要 class？

一个线性模型有两类东西：

```text
数据：weights 和 bias。
能力：predict() 和 predict_batch()。
```

`class` 可以把它们放在同一个地方：

```python
class LinearModel:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, features):
        ...
```

之后创建模型：

```python
model = LinearModel([5, 3, -4], 40)
```

此时 `model` 里保存了：

```text
model.weights = [5, 3, -4]
model.bias = 40
```

## 2. `self` 到底是什么？

`self` 表示“当前这个模型对象”。

当你写：

```python
model = LinearModel([5, 3, -4], 40)
```

在 `__init__()` 中的 `self` 就是这个 `model`。

所以：

```python
self.weights = weights
```

等价于把传进来的权重保存到：

```python
model.weights
```

同一个类可以创建不同参数的多个模型：

```python
model_a = LinearModel([5, 3, -4], 40)
model_b = LinearModel([2, 1, 0], 10)
```

它们用同一套预测规则，但各自保存不同的参数。

## 3. 单样本预测

`predict()` 接收一个样本的特征：

```python
features = [2, 5, 0]
```

然后使用模型自己的参数完成计算：

```python
prediction = model.predict(features)
```

这里不再需要写：

```python
multi_linear(features, weights, bias)
```

因为 `weights` 和 `bias` 已经在 `model` 内部了。

## 4. 批量预测

一个模型不仅可以预测一个样本，也可以预测很多样本：

```python
samples = [
    [2, 5, 0],
    [4, 4, 1],
    [1, 2, 3],
]

predictions = model.predict_batch(samples)
```

`predict_batch()` 的内部就是：

```text
依次取出每个样本
-> 调用同一个模型自己的 predict()
-> 收集每个预测结果
```

这样做有一个好处：

```text
单样本计算只写在 predict() 里。
批量计算重复调用 predict()。
```

这叫作复用已有逻辑。以后代码变复杂时，它能显著减少错误。

## 5. 参数更新意味着什么？

今天会添加一个 `set_parameters()` 方法：

```python
model.set_parameters([6, 2, -3], 35)
```

它会把模型内部的参数替换成新值。

现在这是手动修改参数；但后面学习梯度下降时，模型会根据预测误差自动更新参数。

所以可以先记住：

```text
训练 = 不断更新模型参数，让预测变得更好。
```

## 6. 与 PyTorch 的联系

以后你会经常写：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x):
        ...
```

今天的 `LinearModel` 和它非常像：

```text
__init__：保存模型需要的内容。
predict：定义输入如何变成输出。
predict_batch：处理多个输入。
```

只是 PyTorch 会替我们管理 Tensor、参数和自动求导。

## 7. 今日关键词

```text
class：类，定义一类对象的结构。
object / instance：对象 / 实例，例如 model。
attribute：属性，例如 model.weights。
method：方法，例如 model.predict()。
initialize：初始化，在 __init__ 中保存初始数据。
```
