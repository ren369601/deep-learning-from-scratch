# Week 6 / Day 6：残差块与配对实验写法

## 1. Identity：不修改数据的模块

```python
self.shortcut = nn.Identity()
```

调用后返回原输入，没有权重。它让恒等捷径与投影捷径都能通过同一个接口 `self.shortcut(images)` 使用。

## 2. 投影捷径

```python
self.shortcut = nn.Sequential(
    nn.Conv2d(8, 16, 1, stride=2, bias=False),
    nn.BatchNorm2d(16),
)
```

把 `(N,8,9,9)` 转成 `(N,16,5,5)`，同时改变通道和空间大小。不是只给输入随意补几张全零特征图。

## 3. 残差相加

```python
features = self.main(images)
shortcut = self.shortcut(images)
assert features.shape == shortcut.shape
return self.relu(features + shortcut)
```

先核对完整形状，再逐位置求和。`torch.cat` 是拼接，不能代替这里的加法。这里不用原地 `+=`，便于清楚追踪计算图中的两条路线。

## 4. 归一化与模式切换

```python
model.train()
loss = loss_fn(model(images), targets)
loss.backward()
optimizer.step()
```

这是训练片段；实际循环还需要在前向前清空梯度。验证则使用：

```python
model.eval()
with torch.no_grad():
    logits = model(images)
```

`eval()` 改变 BatchNorm 等层的行为，但不自动关闭求导；`no_grad()` 关闭求导记录，但不自动切换模型行为。

## 5. 深拷贝相同初始状态

```python
from copy import deepcopy

plain = TinyClassifier(use_skip=False)
residual = deepcopy(plain)
for block in residual.blocks:
    block.use_skip = True
```

复制模型后，两者拥有数值相同、但彼此独立的参数与缓冲区。训练其中一组不会同时更新另一组。

`residual = plain` 只是同一个对象的两个名字，不是复制，不能用来做独立训练对照。

## 6. state_dict 核对什么

```python
for name, value in plain.state_dict().items():
    assert torch.equal(value, residual.state_dict()[name])
```

它包括权重和 BatchNorm 的运行统计等缓冲区，不仅是 `parameters()`。本例的 `use_skip` 是普通布尔属性，不在这个字典里，已经在前面的循环中显式切换。

## 7. 检查梯度，不从输出直接读 grad

```python
images = torch.randn(2, 8, 8, 8, requires_grad=True)
output = block(images)
output.square().mean().backward()
assert images.grad is not None
```

`images` 是主动要求梯度的叶子张量；`output` 是中间结果，默认不保留 `.grad`。今天只检查输入与参数的梯度，无需调用 `retain_grad()`。

## 8. 数据批次顺序复现

```python
loader = DataLoader(dataset, batch_size=16, shuffle=True,
                    generator=torch.Generator().manual_seed(202))
```

两次独立训练各创建自己的生成器，但用相同种子，可得到一致的打乱序列。不要让两组训练共用一个已被前一组消耗的生成器。

种子有助于在同一环境重复观察；不承诺不同硬件和版本得到逐位相同的结果。

## 9. 运行完整作业

在项目根目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 week06/day06_resnet_lab.py
```

文件依次运行两个示例、两个已完成练习和六次训练（三个种子×两种网络）。仅使用本地合成数据，不下载数据或预训练权重。

它复用同目录 Day 3 的生成与评估函数，因此保留 `day03_cnn_classifier.py`。导入只定义这些函数，不会触发 Day 3 底部的训练入口。
