# Week 5 / Day 5：小批量、优化器与预处理写法

## 1. TensorDataset 与 DataLoader

~~~python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

X = torch.arange(20, dtype=torch.float32).reshape(10, 2)
y = torch.arange(10, dtype=torch.float32).reshape(10, 1)
dataset = TensorDataset(X, y)
generator = torch.Generator().manual_seed(202)
loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)
for X_batch, y_batch in loader:
    print(X_batch.shape, y_batch.shape)
~~~

一轮的三个 batch 大小是 4、4、2。shuffle 改变样本顺序，但输入与标签保持对应。默认不丢弃末尾不满 batch 的样本。

## 2. 只用训练输入拟合标准化

~~~python
X_train = torch.tensor([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])
X_val = torch.tensor([[25.0, 2.5]])
mean = X_train.mean(dim=0, keepdim=True)
std = X_train.std(dim=0, correction=0, keepdim=True).clamp_min(1e-6)
train_scaled = (X_train - mean) / std
val_scaled = (X_val - mean) / std
print(mean.shape, std.shape, train_scaled.mean(dim=0), val_scaled)
~~~

dim=0 沿样本求统计量，保留各列特征；keepdim=True 保持 (1,D)；clamp_min 给标准差设置下限。

## 3. 不同优化器的创建

~~~python
model = nn.Linear(2, 1)
sgd = torch.optim.SGD(model.parameters(), lr=0.03)
momentum = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
adam = torch.optim.Adam(model.parameters(), lr=0.03)
~~~

这里仅展示接口。真正做实验时，每个优化器配一个重新创建的模型，不能在同一个已训练模型上依次更新来比较。

## 4. 小批量训练的一步

~~~python
model.train()
loss_fn = nn.BCEWithLogitsLoss()
X_batch = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
y_batch = torch.tensor([[1.0], [0.0]])
sgd.zero_grad()
logits = model(X_batch)
loss = loss_fn(logits, y_batch)
loss.backward()
sgd.step()
~~~

把这一段放进 for X_batch,y_batch in loader 的循环，才能完整训练一轮。每个 batch 清梯度并更新一次。

## 5. 初始化写法（参考）

~~~python
hidden = nn.Linear(2, 8)
nn.init.kaiming_normal_(hidden.weight, mode="fan_in", nonlinearity="relu")
nn.init.zeros_(hidden.bias)
~~~

Kaiming 这里用于随后接 ReLU 的隐藏层。不要无差别地对所有层套用同一种假设。nn.init 的初始化操作内部在不记录梯度的条件下执行。

## 6. 学习率调度（参考）

~~~python
scheduler = torch.optim.lr_scheduler.StepLR(sgd, step_size=10, gamma=0.5)
~~~

按轮调度时，在这一轮所有 optimizer.step() 执行完后调用 scheduler.step()。今天固定学习率的对照不使用调度器。

## 7. 日志与实验函数

~~~python
print("current learning rate:", sgd.param_groups[0]["lr"])
print("finite loss:", torch.isfinite(loss).item())
~~~

今日提供 run_experiment(kind, learning_rate, epochs=50)，kind 可为 "sgd"、"momentum"、"adam"。返回每轮的 history，包含 epoch、train_loss、train_acc、val_loss、val_acc、updates。

history[0] 对应更新前，history[10] 对应第 10 轮更新后，history[50] 对应第 50 轮更新后。不要把 epoch=0 当作做了一轮训练。

在虚拟环境中运行 python week05/day05_optimization.py。示例已写好；两道练习与作业由你完成。
