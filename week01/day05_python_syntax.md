# Week 1 / Day 5：需要掌握的 Python 语法

今天新增的重点是比较大小和 `if`。它们让程序能够保留“目前最好的模型”。

## 1. 比较运算符

```python
3 < 5   # True，小于
3 > 5   # False，大于
3 == 3  # True，是否相等
3 != 5  # True，是否不相等
```

模型选择时最常用的是：

```python
loss < best_loss
```

意思是：当前模型的损失是否比目前记录的最优损失更小。

## 2. `if` 条件判断

```python
score = 90

if score > 60:
    print("通过")
```

当条件为 `True` 时，才执行缩进中的代码。

在寻找最优模型时：

```python
if loss < best_loss:
    best_loss = loss
    best_weight = weight
    best_bias = bias
```

它表示：

```text
如果发现更小的 loss，
就用当前参数和当前 loss 替换之前记录的最优结果。
```

## 3. 双层循环

为了尝试多组权重和偏置，可以写两层循环：

```python
candidate_weights = [0, 1, 2, 3]
candidate_biases = [0, 1, 2]

for weight in candidate_weights:
    for bias in candidate_biases:
        print(weight, bias)
```

它会依次尝试所有组合：

```text
(0, 0), (0, 1), (0, 2)
(1, 0), (1, 1), (1, 2)
(2, 0), (2, 1), (2, 2)
(3, 0), (3, 1), (3, 2)
```

这正好可以用于尝试不同的模型参数。

## 4. `float("inf")`

寻找最小 loss 时，初始值可以写成：

```python
best_loss = float("inf")
```

`inf` 表示无穷大。

因为任何正常计算出的 loss 都比无穷大更小，所以第一个候选模型一定会成为当前最优模型。

## 5. 今天最重要的组合

```python
best_loss = float("inf")

for weight in candidate_weights:
    for bias in candidate_biases:
        loss = mean_squared_error(xs, ys, weight, bias)

        if loss < best_loss:
            best_loss = loss
            best_weight = weight
            best_bias = bias
```

请能解释：为什么 `best_loss` 一开始要很大，以及什么情况下会更新最优参数。
