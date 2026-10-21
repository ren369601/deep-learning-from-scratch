# Week 6 / Day 5：网络结构与分支的 PyTorch 写法

## 1. Sequential：按顺序执行

```python
from torch import nn

features = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
)
```

输入先经过卷积，再经过激活，最后池化。`Sequential` 适合一条直线的数据流；四条并行分支应分别定义，再在 `forward` 中组合。

## 2. named_children：查看直接子层

```python
for name, layer in features.named_children():
    images = layer(images)
    print(name, images.shape)
```

每次把当前结果交给下一层，适合追踪形状。如果是嵌套容器，这里只访问第一层子模块；不是递归列出所有最底层算子。

## 3. numel 与 parameters：核对参数总数

```python
count = sum(parameter.numel() for parameter in model.parameters())
```

`parameters()` 给出模型注册的参数；`numel()` 给出某个参数张量中的数值数量。这个表达式把各个参数张量的数量相加，包括默认存在的偏置。今天所有参数都可训练。

`image.numel()` 则是在数输入里的数值，不是数模型参数。

## 4. 四条分支共享输入

```python
def forward(self, images):
    first = self.branch1(images)
    second = self.branch2(images)
    third = self.branch3(images)
    fourth = self.branch4(images)
    return torch.cat([first, second, third, fourth], dim=1)
```

四个调用都传入 `images`，不能写成把第一条分支的结果传给第二条。`dim=1` 是通道维；`dim=0` 会沿批量维拼接，不符合今天的目标。

这些分支必须在 `__init__` 中赋给 `self.branch1` 等属性，才能被模型正确注册。不要在每次 `forward` 时重新创建卷积层。

## 5. 保持空间大小

```python
nn.Conv2d(4, 6, 3, padding=1)
nn.Conv2d(2, 2, 5, padding=2)
nn.MaxPool2d(3, stride=1, padding=1)
```

这里步长为 1，卷积的空洞率为 1，三者都保持输入高宽。池化若省略步长，默认会使用核大小，导致意外缩小，所以这里明确写 `stride=1`。

## 6. 全局平均池化与展平

```python
pool = nn.AdaptiveAvgPool2d((1, 1))
flatten = nn.Flatten(1)
pooled = pool(features)
vectors = flatten(pooled)
```

若特征为 `(2,16,16,16)`，则 `pooled` 为 `(2,16,1,1)`，`vectors` 为 `(2,16)`。后面接 `Linear(16,3)`，不是 `Linear(4096,3)`。

## 7. 前向检查不等于训练

```python
model.eval()
with torch.no_grad():
    logits = model(images)
```

`eval()` 切换模型行为，`no_grad()` 关闭这段代码的梯度记录。二者不能互相替代。今天示例只做前向检查；真正训练时不用把前向包在 `no_grad()` 中。

## 8. 三分类损失

```python
targets = torch.tensor([0, 2], dtype=torch.long)
loss = nn.CrossEntropyLoss()(logits, targets)
```

两张输入对应两个整数标签，标签范围是 0～2；`logits.shape` 应为 `(2,3)`。标签不是三个类别分数，也不需要先做 one-hot。损失能计算只说明形状与数据类型兼容，不说明模型已经学会分类。

## 9. 今日运行

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 week06/day05_cnn_architectures.py
```

刚打开时只有示例已完成；练习和作业会显示待完成提示，这是预留给你填写的部分，不是运行故障。
