# Week 5 / Day 4：前馈网络与三折验证写法

## 1. 手写前向与 Linear 参数对齐

~~~python
import torch
from torch import nn

X = torch.tensor([[1.0, 2.0]])
W = torch.tensor([[1.0, -1.0], [2.0, 0.5]])
b = torch.tensor([0.0, 1.0])
manual = X @ W + b
layer = nn.Linear(2, 2)
with torch.no_grad():
    layer.weight.copy_(W.T)
    layer.bias.copy_(b)
print(torch.allclose(manual, layer(X)))
~~~

copy_ 复制数值到已注册参数；不要为了替换数值而随意覆盖 Parameter 对象。W 采用 (输入,输出)，Linear 的 weight 则采用 (输出,输入)。

## 2. 两层分类网络

~~~python
model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
X = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
targets = torch.tensor([1, 1], dtype=torch.long)
loss = nn.CrossEntropyLoss()(model(X), targets)
loss.backward()
for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.grad.shape)
~~~

Sequential 按顺序调用每个模块。若要更自由地组织 forward，可写 nn.Module 类。上面隐藏宽度为 4，今天作业的 MLP 使用 16。

## 3. 用索引取数据

~~~python
X = torch.arange(12, dtype=torch.float32).reshape(6, 2)
targets = torch.tensor([0, 0, 0, 1, 1, 1])
train_idx = torch.tensor([0, 2, 3, 5])
val_idx = torch.tensor([1, 4])
X_train, y_train = X[train_idx], targets[train_idx]
X_val, y_val = X[val_idx], targets[val_idx]
print(X_train.shape, X_val.shape)
~~~

输入和标签必须使用同一组索引。验证索引不应出现在该折训练索引中。

## 4. 今日提供的函数

make_xor_data() 返回 X (48,2)、y (48,)；每个 XOR 角附近有 12 个带小扰动的样本。

make_stratified_folds(y, k=3, seed=13) 返回三组验证索引，每组包含相同比例的两个类别。要构造某折训练索引，就合并另外两组。

fit_fold(kind, X_train, y_train, X_val, y_val, seed) 每次创建新模型与优化器，训练 200 轮，然后返回四个浮点指标：

~~~text
train_loss, train_acc, val_loss, val_acc
~~~

kind 只能填 "linear" 或 "mlp"。调用前自己组织好每折数据，不能用前一折训练得到的权重。

## 5. 汇总数值

~~~python
scores = torch.tensor([0.75, 0.875, 1.0])
print("mean:", scores.mean().item())
print("population std:", scores.std(correction=0).item())
~~~

correction=0 使用总体标准差定义；默认的样本标准差除数不同。mean/std 都只描述本次各折分数，不自动提供统计显著性结论。

## 6. 运行和记录

在虚拟环境中运行 python week05/day04_mlp_model_selection.py。示例可以直接运行；练习和作业打印提示表示相应部分还要自己补齐。

作业注释中写明数据 seed=7、折划分 seed=13、模型种子按 42+fold_id 设置（fold_id 从 0 开始），以及训练轮数与学习率。记录每折成绩后，再描述均值和波动。
