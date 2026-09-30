# Week 1 / Day 2：多输入线性模型

Day 1 的模型只有一个输入：

```text
y = w * x + b
```

但真实问题通常不会只看一个信息。

例如预测一套房子的价格时，可能同时参考：

```text
x1：面积
x2：房间数量
x3：房龄
```

这时模型可以写成：

```text
y = w1 * x1 + w2 * x2 + w3 * x3 + b
```

这就是**多输入线性模型**，也可以叫作多特征线性模型。

## 1. 什么是特征？

`feature`（特征）就是模型用来做预测的输入信息。

在房价例子中：

```text
面积、房间数量、房龄，都是特征。
```

一个样本的特征可以写成一个 list：

```python
features = [80, 2, 10]
```

它表示：

```text
面积 80
房间数 2
房龄 10
```

模型也需要为每个特征准备一个权重：

```python
weights = [0.8, 15, -1.2]
```

含义是：

```text
面积每增加 1 单位，预测值增加 0.8
房间数每增加 1 个，预测值增加 15
房龄每增加 1 年，预测值减少 1.2
```

这里的数值只是为了理解，不代表真实房价模型。

## 2. 模型如何计算？

给定：

```python
features = [80, 2, 10]
weights = [0.8, 15, -1.2]
bias = 20
```

模型会把位置对应的特征和权重相乘，再全部相加：

```text
y = 80 * 0.8 + 2 * 15 + 10 * (-1.2) + 20
  = 64 + 30 - 12 + 20
  = 102
```

重点是“位置对应”：

```text
features[0] <-> weights[0]  # 面积对应面积的权重
features[1] <-> weights[1]  # 房间数对应房间数的权重
features[2] <-> weights[2]  # 房龄对应房龄的权重
```

如果顺序错了，模型的含义就错了。

## 3. 点积：今天先认识，不必死记

下面这部分求和：

```text
w1 * x1 + w2 * x2 + w3 * x3
```

叫作**点积**（dot product）。

可以简写为：

```text
w · x
```

所以多输入线性模型还可以写成：

```text
y = w · x + b
```

其中：

```text
x = [x1, x2, x3]  # 特征
w = [w1, w2, w3]  # 权重
```

现在只需要理解：**点积就是对应元素相乘后求和。**

NumPy 和 PyTorch 后面会帮我们高效完成这个计算，但理解它的原始写法很重要。

## 4. 用 Python 实现

最直接的写法是用循环：

```python
def multi_linear(features, weights, bias):
    total = 0

    for feature, weight in zip(features, weights):
        total += feature * weight

    return total + bias
```

这里：

```text
total：暂时保存累计结果
zip(features, weights)：每次取出一对对应的 feature 和 weight
total += feature * weight：等价于 total = total + feature * weight
```

## 5. 为什么这和神经网络直接相关？

以后会写到：

```python
nn.Linear(in_features=3, out_features=1)
```

它要做的核心计算仍然是：

```text
y = w1 * x1 + w2 * x2 + w3 * x3 + b
```

区别只在于：

```text
今天：一个样本、一个输出，用 Python list 和 for 循环手写。
以后：一批样本、多个输出，用矩阵和 PyTorch 高效计算。
```

底层逻辑没有变。

## 6. 今天要形成的直觉

```text
特征 feature：模型参考的输入信息。
权重 weight：每个特征的重要程度和影响方向。
偏置 bias：不依赖输入的基础调整量。
预测 prediction：所有“特征 × 对应权重”的和，再加 bias。
```

请能够用自己的话解释：

```text
为什么 features 和 weights 的长度必须相同？
为什么 x1 必须乘 w1，而不能随便交换？
```

## 7. 今日关键词

```text
feature：特征
multiple features：多个特征
weight：权重
bias：偏置
dot product：点积
prediction：预测
```
