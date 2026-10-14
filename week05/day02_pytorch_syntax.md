# Week 5 / Day 2：分类模型的 PyTorch 写法

本页是详细知识讲解的代码对照。今天最重要的是分清“原始分数”“概率”“类别”以及两种损失需要的标签格式。

## 1. 一个 logit 的二分类

~~~python
import torch
from torch import nn

X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # (N=2,D=2)
targets = torch.tensor([[1.0], [0.0]])         # (N,1)，浮点数
model = nn.Linear(2, 1)
logits = model(X)                             # (N,1)，任意实数
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, targets)               # 标量
~~~

targets 的元素取 0.0 或 1.0；shape 必须与 logits 一致。这里一个 logit 表示对类别 1 的支持程度，类别 0 的概率由 1-p 得到。

![image-20261012211536544](C:\Users\任博宇\AppData\Roaming\Typora\typora-user-images\image-20261012211536544.png)

## 2. 概率和类别只在评价时转换

~~~python
with torch.no_grad():
    probabilities = torch.sigmoid(logits)
    predicted = (probabilities >= 0.5).float()
    accuracy = (predicted == targets).float().mean().item()
~~~

sigmoid 保持 shape 不变。>= 生成布尔 Tensor；float() 变成 0/1，方便与目标比较。对单输出二分类不要用 argmax(dim=1)：这一维只有一个元素，结果永远是索引 0。

## 3. 完成一次参数更新

~~~python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
model.train()
optimizer.zero_grad()
loss = loss_fn(model(X), targets)
loss.backward()
optimizer.step()
~~~

训练使用原始 logits，Sigmoid 已包含在损失的稳定计算中。不能把 sigmoid(logits) 交给 BCEWithLogitsLoss；需要处理概率输入的其他损失不是今天的训练写法。

## 4. C 个 logits 的多分类

~~~python
logits = torch.tensor([[2.0, 1.0, 0.0]])  # (N=1,C=3)
targets = torch.tensor([0], dtype=torch.long)  # (N,)
loss = nn.CrossEntropyLoss()(logits, targets)
probabilities = torch.softmax(logits, dim=1)
predicted = logits.argmax(dim=1)
~~~

dim=1 表示沿类别维度归一化或选最大值。targets 保存类别编号，不能写成 (N,1) 的浮点 0/1 标签。今天不涉及软标签。

| 任务写法 | logits shape | 标签 shape/dtype | 训练损失 | 预测类别 |
|---|---|---|---|---|
| 一个输出二分类 | (N,1) | (N,1)，float32 | BCEWithLogitsLoss | sigmoid 后比较阈值 |
| C 个输出多分类 | (N,C) | (N,)，long | CrossEntropyLoss | argmax(dim=1) |

## 5. 多数类别基线

~~~python
train_labels = torch.tensor([[0.0], [0.0], [1.0]])
majority_class = (train_labels.mean() >= 0.5).float()
val_labels = torch.tensor([[0.0], [1.0]])
baseline_predictions = torch.full_like(val_labels, majority_class.item())
baseline_accuracy = (baseline_predictions == val_labels).float().mean().item()
~~~

基线类别只由训练标签决定；验证标签用于计算准确率。这里约定平票时选 1。换个约定也可以，但要固定并写清楚。

## 6. 手写例子与库函数的边界

理论上二元交叉熵可以写成 -[y*log(p)+(1-y)*log(1-p)]。手算小例子能帮助理解；实际训练直接用 BCEWithLogitsLoss，避免 p 接近 0 或 1 时的数值问题。

今天的示例还有一个感知机单步更新，用 -1/+1 标签；不要把它与 BCE 的 0/1 标签混用。

运行方式：在仓库根目录使用 .venv/Scripts/python.exe week05/day02_classification_baselines.py，或在已激活虚拟环境的终端执行 python week05/day02_classification_baselines.py。

参考：[BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.loss.BCEWithLogitsLoss.html)、[CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)。
