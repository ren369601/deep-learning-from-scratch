# Week 3 / Day 3：Cross Entropy 所需 NumPy 语法

## 1. `np.log()`

```python
probabilities = np.array([0.8, 0.5, 0.1])
negative_logs = -np.log(probabilities)
```

概率越小，负对数越大。

## 2. `np.clip()`

```python
safe = np.clip(probabilities, 1e-12, 1.0)
```

防止出现 `log(0)`。

## 3. 高级索引

```python
probabilities = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7],
])
targets = np.array([0, 1, 2])

rows = np.arange(len(targets))
correct = probabilities[rows, targets]
```

结果：

```text
[0.8, 0.7, 0.7]
```

## 4. `np.argmax()`

```python
predicted_classes = np.argmax(probabilities, axis=1)
```

对每一行找出最大概率所在的列下标。

## 5. 布尔值求平均

```python
correct = predicted_classes == targets
accuracy = np.mean(correct)
```

NumPy 会把：

```text
True 当作 1
False 当作 0
```

所以平均值就是正确比例。

## 6. One-Hot

```python
targets = np.array([0, 2, 1])
one_hot = np.eye(3)[targets]
```

结果：

```text
[[1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]]
```

## 7. Cross Entropy 函数结构

```python
def cross_entropy(probabilities, targets):
    safe = np.clip(probabilities, 1e-12, 1.0)
    rows = np.arange(len(targets))
    correct_probabilities = safe[rows, targets]
    return float(-np.mean(np.log(correct_probabilities)))
```
