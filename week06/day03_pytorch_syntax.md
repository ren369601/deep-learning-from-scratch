# Week 6 / Day 3：CNN 类、展平与分类训练写法

## 1. 用 Sequential 观察一小段网络

~~~python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

features = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
)
X = torch.zeros((2, 1, 8, 8))
hidden = features(X)
print(hidden.shape)
~~~

结果为 (2,4,4,4)。接下来再做4到8通道的卷积和一次池化，才会变成 (2,8,2,2)。

## 2. 展平时保留batch

~~~python
hidden = torch.zeros((2, 8, 2, 2))
flattened = torch.flatten(hidden, start_dim=1)
print(flattened.shape)
classifier = nn.Linear(32, 2)
print(classifier(flattened).shape)
~~~

结果分别为 (2,32)、(2,2)。start_dim=1 保留样本轴。也可以在模块中使用 nn.Flatten(1)。

## 3. 模型类的基本骨架

~~~python
class ExampleBlock(nn.Module):
    """示例：把一个卷积块注册成模型的一部分。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, X):
        return self.features(X)
~~~

super().__init__() 初始化模块管理功能；self.features 注册子模块；forward 描述数据流。今日 TinyCNN 需要自己再接上池化、第二层和分类器。

## 4. 一次分类更新

~~~python
model = nn.Sequential(nn.Flatten(1), nn.Linear(64, 2))
X_batch = torch.zeros((2, 1, 8, 8))
y_batch = torch.tensor([0, 1], dtype=torch.long)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
model.train()
optimizer.zero_grad()
loss = loss_fn(model(X_batch), y_batch)
loss.backward()
optimizer.step()
~~~

这段用最小线性分类器展示更新写法。今日作业换成你完成的 TinyCNN，训练流程相同。

## 5. 观察卷积核的梯度与更新

~~~python
conv = nn.Conv2d(1, 4, 3, padding=1)
weights_before = conv.weight.detach().clone()
loss = conv(torch.ones(2, 1, 8, 8)).square().mean()
loss.backward()
print(conv.weight.grad.shape)
~~~

比较更新前后参数时，必须在 optimizer.step() 前保存独立快照，再在 step 后比较。backward 本身不会改变参数数值。

## 6. 已提供的辅助函数

make_line_images(per_class=64, seed=7) 返回 images (128,1,8,8) 与 targets (128,)。

evaluate_classifier(model, loader) 返回全数据平均交叉熵与准确率。它按样本数累计损失，避免末尾小batch被等权放大。

build_reference_cnn() 是示例使用的完整网络；练习与作业使用你自己完成的 TinyCNN。

验证集用 per_class=32、seed=17 单独生成，不能把训练 loader 当作验证 loader。多数类基线只从训练标签选类别，本次平票时可以固定选0。

运行 python week06/day03_cnn_classifier.py。模型类和练习作业仍留有待完成位置。
