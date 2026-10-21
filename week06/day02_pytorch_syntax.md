# Week 6 / Day 2：通道、参数量与池化写法

## 1. 把两个通道堆成同一个样本

~~~python
import torch
from torch import nn
from torch.nn import functional as F

first = torch.arange(1,10,dtype=torch.float32).reshape(3,3)
image = torch.stack([first,10*first], dim=0)
batch = image.unsqueeze(0)
print(image.shape, batch.shape)  # (2,3,3), (1,2,3,3)
~~~

stack 增加一个轴，把两张二维表作为通道；unsqueeze 再增加样本轴。输入已有通道时，不要再把通道数误当样本数。

## 2. 一个输出通道的两路贡献

~~~python
kernels = torch.tensor([[[1.0,0.0],[0.0,-1.0]],
                        [[0.0,1.0],[1.0,0.0]]])
contributions = (image[:,:2,:2] * kernels).sum(dim=(1,2))
output_value = contributions.sum() + 1.0
print(contributions, output_value)
~~~

dim=(1,2) 对每路的核高宽求和，保留两路贡献；再 sum 合成一个输出。偏置1只加一次。

## 3. 创建普通多通道卷积

~~~python
layer = nn.Conv2d(3,4,kernel_size=3,padding=1)
X = torch.zeros((2,3,8,8))
print(layer.weight.shape, layer.bias.shape)
print(layer(X).shape)
print("parameters:", sum(p.numel() for p in layer.parameters()))
~~~

weight=(4,3,3,3)，输出=(2,4,8,8)，参数数112。N=2 没有改变参数数。

## 4. 池化分别处理每个通道

~~~python
values = torch.tensor([[1.0,3.0,2.0,0.0],
                       [4.0,6.0,5.0,1.0],
                       [0.0,2.0,8.0,4.0],
                       [1.0,3.0,6.0,2.0]])
max_output = F.max_pool2d(values[None,None],kernel_size=2,stride=2)
avg_output = F.avg_pool2d(values[None,None],kernel_size=2,stride=2)
print(max_output[0,0], avg_output[0,0])
~~~

也可以使用 nn.MaxPool2d(2,stride=2) 和 nn.AvgPool2d(2,stride=2) 模块。这里没有 padding。

## 5. 在手写代码中对通道求和

~~~text
for 每个输出通道:
    初始化一张输出图为0
    for 每个输入通道:
        使用这个输入通道对应的二维核
        调用昨天的 manual_corr2d
        把贡献加到输出图
    最后加一次当前输出通道的偏置
~~~

weight[out_channel,in_channel] 取出一张二维核。image[in_channel] 取出对应二维输入。

今天参考检查只对照 F.conv2d。手写函数内部复用昨天的实现，不能直接用库卷积替代通道组合。

## 6. 看懂平移示例

shift_right 把输入最后一列舍弃、左侧补0，将其余元素向右移一格。它与循环回绕的 torch.roll 有不同的边界约定。

示例使用远离边界的亮点，核对卷积响应是否一起移动；池化例子则观察亮点是否跨过窗口分界。

运行 python week06/day02_cnn_principles.py。示例已完成，练习和作业仍需你填写。
