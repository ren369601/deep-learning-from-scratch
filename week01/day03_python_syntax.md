# Week 1 / Day 3：需要掌握的 Python 语法

今天新增三个小知识点：嵌套 list、`append()` 和 `enumerate()`。

## 1. 嵌套 list

list 里的每个元素也可以是 list：

```python
samples = [
    [60, 1, 20],
    [80, 2, 10],
    [100, 3, 5],
]
```

这叫嵌套 list。

访问方式：

```python
print(samples[0])
print(samples[0][1])
```

输出：

```text
[60, 1, 20]
1
```

解释：

```text
samples[0]：第一个样本。
samples[0][1]：第一个样本中的第二个特征。
```

## 2. `append()`：向 list 末尾加入元素

一开始创建一个空 list：

```python
predictions = []
```

每得到一个预测结果，就加入 list：

```python
predictions.append(59.0)
predictions.append(102.0)
```

此时：

```python
predictions == [59.0, 102.0]
```

在循环中经常这样用：

```python
results = []

for number in [1, 2, 3]:
    results.append(number * 10)

print(results)
```

输出：

```text
[10, 20, 30]
```

## 3. `enumerate()`：同时得到位置和元素

如果想打印“第几个样本”，可以使用 `enumerate()`：

```python
names = ["小明", "小红", "小刚"]

for index, name in enumerate(names, start=1):
    print(index, name)
```

输出：

```text
1 小明
2 小红
3 小刚
```

其中：

```text
index：当前位置。
name：当前位置的元素。
start=1：让编号从 1 开始，而不是 Python 默认的 0。
```

今天的代码会用它打印每个样本的预测结果。

## 4. 今天最重要的组合

```python
predictions = []

for features in samples:
    prediction = multi_linear(features, weights, bias)
    predictions.append(prediction)
```

请能逐句解释：

```text
先创建一个空 list 保存结果。
从 samples 中依次取出每个样本。
预测当前样本。
把当前预测结果加入 predictions。
```
