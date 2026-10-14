# Week 5 / Day 3：转换、参数与分类指标写法

本页配合详细知识讲解使用。重点看每一步是复制数据、计算梯度，还是只记录结果。

## 1. NumPy 转 Tensor

~~~python
import numpy as np
import torch
from torch import nn

array = np.array([[1.0, 2.0]], dtype=np.float32)
shared = torch.from_numpy(array)
copied = torch.tensor(array)
array[0, 0] = 9.0
print(shared, copied)
~~~

这个例子里 shared 的第一项变成 9，copied 保持 1。独立保存数组可使用 copy；独立保存 Tensor 可使用 clone。

## 2. 回到 NumPy 与检查设备

~~~python
value = torch.tensor([2.0], requires_grad=True)
numpy_snapshot = value.detach().cpu().numpy().copy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, numpy_snapshot)
~~~

detach 去掉求导关系，cpu 负责设备，末尾 copy 负责独立内存。仅记录一个标量可用 value.item()。

## 3. 手动参数的一次更新

~~~python
X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
y = torch.tensor([[2.0], [-1.0]])
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
loss = nn.MSELoss()(X @ W + b, y)
loss.backward()
with torch.no_grad():
    W -= 0.05 * W.grad
    b -= 0.05 * b.grad
W.grad.zero_()
b.grad.zero_()
~~~

不在循环里重新创建 W、b；每轮只重新前向计算 loss。不要把 W 改写成带计算历史的新 Tensor 后，又期待它像原先的叶子参数一样保存 .grad。

## 4. nn.Module 管理参数

~~~python
class ExampleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, X):
        return self.linear(X)

model = ExampleRegressor()
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
~~~

今天的练习输入为两个特征，模型维度需要相应调整。对照实验的零初始化可在 torch.no_grad() 中调用 model.linear.weight.zero_() 和 model.linear.bias.zero_()；训练优化器使用 model.parameters()。

## 5. 参数维度对齐

~~~python
manual_weights = torch.zeros((3, 1))
module_weights = model.linear.weight.detach().T
print(manual_weights.shape, module_weights.shape)
print(torch.allclose(manual_weights, module_weights))
~~~

这里 model 来自上面的三特征示例；allclose 比较浮点数是否在容差内接近。数值不同可能因为初始化不同，先确认条件再比较结果。

## 6. 一个分类计数例子

~~~python
targets = torch.tensor([1, 1, 0])
predicted = torch.tensor([1, 0, 0])
tp = ((targets == 1) & (predicted == 1)).sum().item()
print("TP:", tp)
~~~

& 是两个布尔 Tensor 的逐元素“与”。其余 TN/FP/FN 改变两个条件即可。输入标签可以先统一成一维，避免 (N,1) 和 (N,) 广播。

~~~python
logits = torch.tensor([[-1.0], [2.0]])
predicted = (torch.sigmoid(logits) >= 0.5).long().reshape(-1)
print(predicted.shape)
~~~

混淆矩阵用整数计数；精确率/召回率用计数比值。当分母为 0，按今天约定返回 0.0。矩阵采用 [[TN, FP], [FN, TP]]，要在输出中说明行列含义。

## 7. 今日运行方式

在虚拟环境中运行 python week05/day03_framework_lab.py。示例可以直接运行；两道练习与作业的 TODO 由你补齐。
