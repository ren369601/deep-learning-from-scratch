# Week 4 / Day 3：用 `nn.Module` 定义模型

Day 1 学了 Tensor，Day 2 学了 Autograd。今天把计算和参数放进一个可复用的模型类。先不训练，也不引入优化器。

## 1. 为什么需要 `nn.Module`

上周手写的模型要自己保存 `W`、`b`，还要自己组织前向计算。继承 `nn.Module` 后，我们可以把层定义在 `__init__()`，把数据流写在 `forward()`，再通过 `model.parameters()` 找到需要学习的参数。

```python
import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, X):
        return self.linear(X)

model = LinearModel(2, 1)
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
output = model(X)  # shape: (2, 1)
```

`super().__init__()` 必须在注册子模块或参数前调用。平时用 `model(X)`，不要直接调用 `model.forward(X)`；前者会经过 PyTorch 模块的正常调用机制。

## 2. `nn.Linear` 与上周的矩阵公式

上周的 NumPy 代码写 `X @ W + b`，`W.shape` 是 `(输入特征数, 输出特征数)`。PyTorch 的 `nn.Linear(D, H)` 在内部保存的 `weight.shape` 是 **`(H, D)`**，`bias.shape` 是 `(H,)`，但输入输出仍是：

```text
X.shape            (N, D)
linear.weight      (H, D)
linear.bias        (H,)
model(X).shape     (N, H)
```

它在内部做的等价计算是 `X @ weight.T + bias`。不要把 `nn.Linear` 的权重 shape 直接当成我们手写的 `W.shape`。

## 3. 参数会自动登记

把 `nn.Linear` 赋给 `self.linear` 后，它的权重和偏置会成为模型的一部分：

```python
for name, parameter in model.named_parameters():
    print(name, parameter.shape)

print(sum(p.numel() for p in model.parameters()))
print(list(model.state_dict().keys()))
```

对 `nn.Linear(3, 2)`，参数量是 `2*3 + 2 = 8`。`state_dict()` 会给出已登记参数的名称和值；今天只查看，不涉及保存文件。

## 4. 自定义可学习参数

不是所有模型都必须使用现成层。若想让某个 Tensor 成为模型参数，可以把 `nn.Parameter` 赋给模型属性：

```python
class ScaleShift(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, X):
        return X * self.scale + self.bias
```

普通 Tensor 属性不会自动列入 `model.parameters()`；`nn.Parameter` 会。今天的作业会用这个机制写一个简单的缩放加偏移模型。

## 5. 与 Autograd 的关系

`nn.Module` 负责组织计算和登记参数；Day 2 的 Autograd 仍负责求梯度。`loss.backward()` 后，可以读 `model.linear.weight.grad` 或遍历 `model.named_parameters()` 查看梯度。**定义模型和求梯度都不会自动更新参数**；Day 4 才学更新。

## 6. 今日完成标准

- 能写出包含 `__init__()` 和 `forward()` 的 `nn.Module` 子类。
- 能解释 `model(X)` 的作用，预测输入输出 shape。
- 能区分 `nn.Linear` 内部 weight 的 shape 与手写 `X @ W` 的 `W.shape`。
- 能列出模型参数，算出参数总数，并让自定义 `nn.Parameter` 获得梯度。

学习顺序：读本页，查 `day03_pytorch_syntax.md`，完成 `day03_nn_module.py`。官方参考：[构建神经网络](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)、[`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)。
