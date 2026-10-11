# Week 4 / Day 5：两层 MLP 用到的 PyTorch 写法

## 1. 定义两层模型

```python
import torch
from torch import nn

class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        hidden = self.relu(self.linear1(X))
        return self.linear2(hidden)  # logits
```

层在 `__init__()` 中登记；数据流在 `forward()` 中写清楚。调用模型用 `model(X)`。

## 2. 查看 shape 与参数总数

```python
model = TwoLayerMLP(2, 8, 2)
for name, parameter in model.named_parameters():
    print(name, parameter.shape)

count = sum(p.numel() for p in model.parameters())
print(count)  # 42
```

## 3. 分类损失与预测

```python
targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
logits = model(X)                           # (4, 2)
loss = nn.CrossEntropyLoss()(logits, targets)  # 标量

probabilities = torch.softmax(logits, dim=1)  # 只用于查看概率
predicted_classes = logits.argmax(dim=1)     # (4,)
```

`CrossEntropyLoss` 接受 logits，**不要**把 `probabilities` 传给它。`dim=1` 表示每个样本沿类别维度归一化。

## 4. 检查梯度

```python
loss.backward()
for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.grad.shape)
```

第一次反向前 `.grad` 可能是 `None`；反向后本题的四组参数都应有梯度。今天只检查，不执行 `optimizer.step()`。
