# Week 2 / Day 2：Reshape、Transpose 与 Broadcasting

昨天我们学习了：

```text
array：数据容器
shape：每个维度的长度
ndim：维度数量
dtype：数据类型
```

今天继续回答三个问题：

```text
怎样在不改变数据总量的情况下改变 shape？
怎样交换行和列？
不同 shape 的数组为什么有时也能一起计算？
```

## 1. Reshape：重新组织数据形状

先创建 6 个数字：

```python
numbers = np.arange(6)
```

它的内容和 shape 是：

```text
[0 1 2 3 4 5]
shape = (6,)
```

可以把它改成 2 行 3 列：

```python
matrix = numbers.reshape(2, 3)
```

结果：

```text
[[0 1 2]
 [3 4 5]]

shape = (2, 3)
```

`reshape` 改变的是数据的组织方式，不是数据本身。

## 2. Reshape 必须保持元素总数一致

原数组有 6 个元素，所以可以变成：

```text
(2, 3)  因为 2 * 3 = 6
(3, 2)  因为 3 * 2 = 6
(1, 6)  因为 1 * 6 = 6
```

但不能变成：

```text
(2, 4)  因为 2 * 4 = 8
```

核心规则：

```text
reshape 前后的元素总数必须相同。
```

## 3. `-1`：让 NumPy 自动推断

```python
numbers = np.arange(12)
matrix = numbers.reshape(3, -1)
```

NumPy 知道一共有 12 个元素，第一维是 3，所以自动推断第二维是 4：

```text
shape = (3, 4)
```

`-1` 只能在一次 `reshape` 中使用一次。

展平数据也可以写：

```python
flat = matrix.reshape(-1)
```

结果重新变成一维，shape 为 `(12,)`。

## 4. Transpose：交换轴

二维矩阵：

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

shape 是：

```text
(2, 3)
```

转置：

```python
transposed = matrix.T
```

结果：

```text
[[1 4]
 [2 5]
 [3 6]]
```

shape 变成：

```text
(3, 2)
```

可以先理解为：行变成列，列变成行。

## 5. Broadcasting：让不同 shape 的数组一起计算

假设有两个样本，每个样本有三个特征：

```python
samples = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

它的 shape 是 `(2, 3)`。

现在给每个特征加上不同的调整量：

```python
feature_bias = np.array([10, 20, 30])
result = samples + feature_bias
```

结果：

```text
[[11 22 33]
 [14 25 36]]
```

虽然两个 shape 不完全相同：

```text
samples.shape      = (2, 3)
feature_bias.shape =    (3,)
```

NumPy 会把 `[10, 20, 30]` 看作应用到每一行。这就是广播。

## 6. 列方向广播

如果想给每一行加不同的数字，可以准备 shape 为 `(2, 1)` 的数组：

```python
row_bonus = np.array([
    [100],
    [200],
])

result = samples + row_bonus
```

结果：

```text
[[101 102 103]
 [204 205 206]]
```

`(2, 1)` 中长度为 1 的列维度会被扩展到 3 列。

## 7. 广播的简化判断规则

从 shape 的最后一个维度开始向前比较。两个维度满足下面任意一个条件，就可以继续广播：

```text
两个维度相等
其中一个维度是 1
其中一个 shape 缺少当前维度
```

例子：

```text
(2, 3) +    (3,)  可以
(2, 3) +  (2, 1)  可以
(2, 3) +    (2,)  不可以
```

最后一个例子中，最后两个维度是 `3` 和 `2`，既不相同，也没有一个是 `1`。

## 8. 与 Transformer 的关系

以后 Multi-Head Attention 会出现类似变换：

```text
(batch, sequence, d_model)
-> reshape
(batch, sequence, heads, head_dim)
-> transpose
(batch, heads, sequence, head_dim)
```

广播也会用于：

```text
给不同位置添加位置编码
把 mask 应用到一批 Attention 分数
给一批数据添加同一组 bias
```

今天学的不是孤立技巧，而是在提前练习 Transformer 的 shape 操作。

## 9. 今日关键词

```text
reshape：改变数据形状
transpose：交换轴
flatten：展平为一维
broadcasting：广播
axis：轴
```
