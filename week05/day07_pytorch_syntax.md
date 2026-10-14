# Week 5 / Day 7：综合实验写法速查

## 1. 数据切分与预处理

从 make_stratified_folds 的索引取训练/验证输入和标签。用训练 X 求 mean/std，再同时变换两份输入；标签保持整数类别编号。

## 2. 小批量分类

~~~python
import copy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

X = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
y = torch.tensor([1, 1], dtype=torch.long)
loader = DataLoader(TensorDataset(X, y), batch_size=2)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 2))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.001)
loss_fn = nn.CrossEntropyLoss()
model.train()
for X_batch, y_batch in loader:
    optimizer.zero_grad()
    loss = loss_fn(model(X_batch), y_batch)
    loss.backward()
    optimizer.step()
~~~

输入 (N,2)，logits (N,2)，标签 (N,)。本次两个输出使用 argmax；不要套用单输出二分类的阈值写法。

## 3. 独立检查点

~~~python
best_state = copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    predicted = model(X).argmax(dim=1)
~~~

在验证改善时保存，停止后恢复。早停最后一轮与恢复后的成绩可能不同。

## 4. 编码混淆矩阵

~~~python
targets = torch.tensor([0, 0, 1, 1])
predicted = torch.tensor([0, 1, 0, 1])
matrix = torch.bincount(2 * targets + predicted, minlength=4).reshape(2, 2)
print(matrix)
~~~

对于 0/1 标签，索引 0、1、2、3 分别代表 TN、FP、FN、TP。矩阵行是真实类别，列是预测类别。此写法仅用于这里的两个类别。

## 5. 运行与复习

运行 python week05/day07_week05_review.py。参考实现已完成，重点结合 day07_knowledge.md 解释流程。运行记录和本周待补事项在 review.md。
