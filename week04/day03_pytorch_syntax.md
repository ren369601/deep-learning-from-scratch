# Week 4 / Day 3：`nn.Module` 用到的 PyTorch 写法

## 1. 定义模型类

```python
from torch import nn

class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, X):
        return self.linear(X)
```

`super().__init__()` 要先调用，再给 `self` 绑定层或参数。层要保存为模型属性，才能被模型登记。

## 2. 实例化与调用

```python
model = MyModel(3, 2)
X = torch.randn(4, 3)
output = model(X)  # (4, 2)
```

`model(X)` 会调用 `forward()`。不要用 `model.forward(X)` 作为日常写法。

## 3. 查看参数和数量

```python
for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.requires_grad)

parameter_count = sum(p.numel() for p in model.parameters())
print(parameter_count)  # nn.Linear(3, 2): 2*3 + 2 = 8
print(list(model.state_dict().keys()))
```

`nn.Linear(3, 2)` 的 `linear.weight.shape` 是 `(2, 3)`，`linear.bias.shape` 是 `(2,)`。本例的 `state_dict` 键为 `linear.weight`、`linear.bias`。

## 4. 登记自定义参数

```python
self.scale = nn.Parameter(torch.tensor([1.0, 2.0]))
self.bias = nn.Parameter(torch.tensor([0.0, -1.0]))
```

这些语句要写在已调用 `super().__init__()` 的模型 `__init__()` 内。`nn.Parameter` 默认需要梯度；普通 `torch.tensor(...)` 赋给 `self` 不会自动登记为参数。

## 5. 检查梯度

```python
output = model(X)
loss = (output ** 2).mean()
loss.backward()

for name, parameter in model.named_parameters():
    print(name, parameter.grad)
```

今天只检查梯度，不修改参数；优化器会在 Day 4 出场。
