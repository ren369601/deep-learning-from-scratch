# Week 5 / Day 6：归一化、Dropout 与早停写法

## 1. 归一化维度

~~~python
import copy
import torch
from torch import nn

X = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
batch_norm = nn.BatchNorm1d(2, affine=False)
layer_norm = nn.LayerNorm(2, elementwise_affine=False)
batch_norm.train()
print(batch_norm(X))  # 按每个特征跨样本归一化
print(layer_norm(X))  # 按每个样本的两个特征归一化
~~~

affine=False 等设置在示例中关闭可学习缩放/偏置，方便观察数值；实际模型通常使用默认的可学习参数。

## 2. Dropout 模式

~~~python
dropout = nn.Dropout(p=0.5)
values = torch.ones(12)
dropout.train()
print(dropout(values))
dropout.eval()
print(dropout(values))
~~~

训练输出由 0 和 2 组成，评估输出都是 1。随机结果不要求每次恰有一半元素为零。

## 3. 权重衰减与显式 L1

~~~python
model = nn.Linear(1, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.03)
X = torch.tensor([[1.0], [2.0]])
y = torch.tensor([[2.0], [4.0]])
data_loss = nn.MSELoss()(model(X), y)
l1_penalty = sum(p.abs().sum() for p in model.parameters())
total_loss = data_loss + 0.001 * l1_penalty
~~~

这里展示两个接口的含义；今天四组对照不同时加入 L1。记录模型预测误差时报告 data_loss，避免把惩罚值混进评价误差。

## 4. 保存独立检查点并恢复

~~~python
best_state = copy.deepcopy(model.state_dict())
with torch.no_grad():
    model.weight.add_(1.0)
model.load_state_dict(best_state)
~~~

这样后续更新不会污染保存的快照。模型若含 BatchNorm，state_dict 还包含运行统计量等缓冲区。若要从检查点继续原训练，还应保存优化器等状态；今天只恢复用于评价的模型。

## 5. 早停逻辑

~~~python
best_loss = float("inf")
bad_epochs = 0
patience = 2
min_delta = 0.005
for epoch, val_loss in enumerate([0.8, 0.6, 0.55, 0.56, 0.58], start=1):
    if val_loss < best_loss - min_delta:
        best_loss = val_loss
        bad_epochs = 0
    else:
        bad_epochs += 1
    if bad_epochs >= patience:
        print("stop epoch:", epoch)
        break
~~~

真正训练还需在改善分支保存状态，循环后恢复。不要把 patience 当作总训练轮数。

## 6. 梯度截断（参考）

~~~python
optimizer.zero_grad()
loss = nn.MSELoss()(model(X), y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
~~~

顺序是 backward -> 截断 -> step，放到 backward 前无效。今天的主要实验不需要添加此操作。

## 7. 今日实验函数

run_regularization(dropout=0.0, weight_decay=0.0, patience=None) 返回结果字典：

- history：逐轮未加惩罚的 train_mse、val_mse，含第 0 轮。
- best_epoch、best_val_mse：满足 min_delta 改善条件的最佳记录。
- stop_epoch：实际完成的最后一轮。
- final_train_mse、final_val_mse：实际选定模型的指标；启用早停时已经恢复检查点。

history 的最后一条是停止前参数的评价，启用早停时不一定等于恢复后的 final 指标。这一点需要在作业解释。

在虚拟环境中运行 python week05/day06_regularization.py。示例可直接运行，练习与作业仍需补齐。
