# Week 1 复习：Python 基础与线性模型

这一周的主线不是记住零散语法，而是逐步搭出一个最小的机器学习流程：

```text
输入数据
-> 模型根据参数进行预测
-> 预测值和真实值比较
-> 计算损失
-> 寻找损失更小的参数
```

## 1. Day 1：单输入线性模型

最简单的线性模型：

```text
y = w * x + b
```

```text
x：输入
w：权重
b：偏置
y：模型输出
```

Python 实现：

```python
def linear(x, weight, bias):
    return weight * x + bias
```

最重要的理解：模型的参数是 `weight` 和 `bias`。参数改变，模型的预测也会改变。

## 2. Day 2：多个特征

真实问题通常有多个输入特征：

```text
y = w1*x1 + w2*x2 + w3*x3 + b
```

```python
features = [x1, x2, x3]
weights = [w1, w2, w3]
```

每个特征必须和同一位置的权重对应：

```python
total = 0

for feature, weight in zip(features, weights):
    total += feature * weight
```

这就是点积的原始形式：对应元素相乘后求和。

## 3. Day 3：批量样本

一个样本是一组特征：

```python
sample = [60, 1, 20]
```

多个样本组成一批数据：

```python
samples = [
    [60, 1, 20],
    [80, 2, 10],
    [100, 3, 5],
]
```

批量预测的核心：

```python
predictions = []

for sample in samples:
    predictions.append(predict(sample))
```

以后会把这种数据理解为 `[batch_size, feature_size]`。

## 4. Day 4：用 class 表示模型

模型既要保存参数，也要提供预测能力：

```python
class LinearModel:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, features):
        ...
```

```text
属性：模型保存的数据，例如 weights、bias。
方法：模型拥有的能力，例如 predict()、predict_batch()。
self：当前模型对象本身。
```

这和以后 PyTorch 的 `nn.Module` 思想一致。

## 5. Day 5：误差与 MSE

只有预测还不够，还需要知道预测得好不好。

单个样本的平方误差：

```text
(prediction - target)^2
```

整个数据集的平均平方误差：

```text
MSE = 所有平方误差之和 / 样本数
```

判断标准：

```text
MSE 越小，当前模型在这组数据上的预测越好。
MSE = 0，表示所有预测都与真实值相同。
```

## 6. 一周知识串联

以数据 `y = 3x + 2` 为例：

```python
xs = [0, 1, 2, 3, 4]
ys = [2, 5, 8, 11, 14]
```

完整过程是：

```text
1. 创建模型，保存 weight 和 bias。
2. 对 xs 中的每个 x 进行预测。
3. 把 predictions 和 ys 一一比较。
4. 计算 MSE。
5. 尝试其他参数，寻找更小的 MSE。
```

当 `weight=3, bias=2` 时，MSE 应为 `0.0`。

## 7. 本周必须能回答的问题

请先不看前几天的文档，尝试口头回答：

1. `weight` 和 `bias` 是什么？
2. 一个样本和一批样本有什么区别？
3. 为什么 `features` 和 `weights` 长度必须相同？
4. `self.weights` 和普通变量 `weights` 有什么区别？
5. `predict_batch()` 为什么适合调用 `self.predict()`？
6. MSE 衡量的是什么？
7. 为什么训练模型要让 loss 变小？
8. 参数不同但输入相同，预测结果为什么可能不同？

## 8. 常见错误

```text
特征和权重数量不一致。
忘记给累计值 total 设置初始值 0。
循环结束前就 return，导致只计算第一个元素。
把 predictions.append(...) 写在循环外面。
创建类属性时漏写 self。
比较模型时误以为 loss 越大越好。
```

## 9. Day 6 综合任务

完成 `day06_week01_review.py`：

```text
1. 写出 LinearRegressionModel。
2. 实现单样本和批量预测。
3. 实现 MSE。
4. 枚举候选参数。
5. 找到 y = 3x + 2 的最佳模型。
```

尽量先凭记忆完成，卡住后再查看 Day 1 到 Day 5 的代码。

## 10. Week 1 完成标准

- [ ] 能独立写出 `y = wx + b`。
- [ ] 能用 `zip()` 计算多个特征的线性输出。
- [ ] 能对嵌套 list 中的多个样本进行预测。
- [ ] 能用 `class` 保存参数并提供预测方法。
- [ ] 能计算 MSE。
- [ ] 能用循环找出候选参数中的最低 MSE。
- [ ] 能解释模型、参数、预测、标签和损失之间的关系。
