# Week 1 / Day 2：需要掌握的 Python 语法

今天新增的语法只有两个：`zip()` 和 `+=`。重点仍然是读懂并写出循环。

## 1. `+=`：累加

```python
total = 0
total += 5
```

这和下面完全一样：

```python
total = 0
total = total + 5
```

所以：

```python
total += feature * weight
```

意思是把当前这一对“特征乘权重”的结果加到 `total` 中。

例子：

```python
total = 0

for number in [2, 3, 4]:
    total += number

print(total)
```

输出：

```text
9
```

过程是：

```text
开始：total = 0
遇到 2：total = 2
遇到 3：total = 5
遇到 4：total = 9
```

## 2. `zip()`：同时遍历两组数据

假设有两个 list：

```python
features = [80, 2, 10]
weights = [0.8, 15, -1.2]
```

`zip(features, weights)` 会把相同位置的元素配成一对：

```text
(80, 0.8)
(2, 15)
(10, -1.2)
```

因此可以这样写：

```python
for feature, weight in zip(features, weights):
    print(feature, weight)
```

输出：

```text
80 0.8
2 15
10 -1.2
```

在循环的每一次：

```text
feature 取一个输入值
weight 取同一位置的权重
```

所以它非常适合计算：

```text
feature * weight
```

## 3. 两个 list 的长度为什么要相同？

```python
features = [80, 2, 10]
weights = [0.8, 15]
```

这代表有 3 个特征，却只有 2 个权重。第三个特征没有对应权重，模型无法正确计算。

今天的代码会主动检查这个问题：

```python
if len(features) != len(weights):
    raise ValueError("features 和 weights 的长度必须相同")
```

先这样理解即可：

```text
if：如果条件成立，就执行缩进里的代码。
len(list)：得到 list 里元素的数量。
raise ValueError：主动提示“输入数据有问题”。
```

这个检查不是模型计算的核心，但它能让代码更可靠。

## 4. 今天最重要的组合

```python
total = 0

for feature, weight in zip(features, weights):
    total += feature * weight

return total + bias
```

你应该能逐句说出它的意思：

```text
先把 total 设为 0。
依次取出一对对应的特征和权重。
将它们相乘，并累加到 total。
循环结束后加上 bias，得到最终预测。
```
