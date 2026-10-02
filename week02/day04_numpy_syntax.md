# Week 2 / Day 4：NumPy 聚合运算与 Axis

## 1. `sum()`

```python
values = np.array([1, 2, 3, 4])

print(values.sum())
print(np.sum(values))
```

两种写法都得到 `10`。

## 2. `mean()`

```python
print(values.mean())
print(np.mean(values))
```

两种写法都得到 `2.5`。

## 3. 平方

```python
errors = np.array([-2, -1, 0, 1, 2])
squared_errors = errors ** 2
```

结果：

```text
[4, 1, 0, 1, 4]
```

## 4. 一行计算 MSE

```python
mse = np.mean((predictions - targets) ** 2)
```

运算顺序：

```text
先相减
再平方
最后求平均
```

## 5. 二维 array 的 axis

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

```python
matrix.mean()        # 所有元素平均，结果是标量
matrix.mean(axis=0)  # 跨行聚合，结果 shape 是 (3,)
matrix.mean(axis=1)  # 每行聚合，结果 shape 是 (2,)
```

记忆方式：指定哪个 `axis`，那个轴就会在结果中消失。

```text
(2, 3) mean(axis=0) -> (3,)
(2, 3) mean(axis=1) -> (2,)
```

## 6. Shape 检查

```python
if predictions.shape != targets.shape:
    raise ValueError("predictions 和 targets 的 shape 必须相同")
```

数组可能因为广播而在错误 shape 下继续计算，所以这个检查很有价值。

## 7. 转成普通 Python 数字

NumPy 计算出的 loss 可能是 NumPy 标量。可以写：

```python
return float(loss)
```

这样函数返回普通 Python `float`，打印和记录更方便。
