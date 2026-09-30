# Week 1 / Day 4：需要掌握的 Python 语法

今天主要复习和强化 `class`、`self`、方法调用，以及一个新概念：对象属性。

## 1. 属性：对象自己的数据

```python
class LinearModel:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
```

`self.weights` 和 `self.bias` 是对象属性。

创建对象后可以读取它们：

```python
model = LinearModel([2, 3], 1)

print(model.weights)
print(model.bias)
```

输出：

```text
[2, 3]
1
```

注意：

```text
weights：__init__ 接收到的临时变量。
self.weights：保存到模型内部、之后还能继续使用的属性。
```

## 2. 方法调用

类里的函数叫方法。

```python
class LinearModel:
    def predict(self, features):
        return 0
```

调用时使用：

```python
model.predict([1, 2])
```

不要手动写 `self`：

```python
# 错误示例
model.predict(model, [1, 2])
```

Python 会自动把 `model` 作为 `self` 传入方法。

## 3. 方法中调用同一对象的其他方法

```python
def predict_batch(self, samples):
    predictions = []

    for features in samples:
        predictions.append(self.predict(features))

    return predictions
```

这里的 `self.predict(features)` 表示：

```text
让当前模型对象预测当前样本。
```

而不是重新写一遍线性计算公式。

## 4. 更新属性

对象属性可以在方法中被修改：

```python
def set_parameters(self, weights, bias):
    self.weights = weights
    self.bias = bias
```

调用前后：

```python
model = LinearModel([2, 3], 1)
print(model.weights)  # [2, 3]

model.set_parameters([4, 5], 0)
print(model.weights)  # [4, 5]
```

## 5. 今天最重要的组合

```python
model = LinearModel(weights=[5, 3, -4], bias=40)

one_prediction = model.predict([2, 5, 0])
batch_predictions = model.predict_batch([
    [2, 5, 0],
    [4, 4, 1],
])
```

你应该能解释：

```text
model 保存了什么？
predict() 接收什么、返回什么？
predict_batch() 为什么能调用 self.predict()？
```
