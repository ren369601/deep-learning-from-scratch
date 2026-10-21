# Week 6 / Day 4：梯度、空洞卷积与转置卷积写法

## 1. 用Autograd查看卷积核梯度

~~~python
import torch
from torch import nn
from torch.nn import functional as F

image = torch.arange(1,10,dtype=torch.float32).reshape(1,1,3,3)
kernel = torch.tensor([[[[1.0,0.0],[0.0,-1.0]]]], requires_grad=True)
loss = F.conv2d(image,kernel).sum()
loss.backward()
print(kernel.grad)
~~~

输出核梯度为 [[12,16],[24,28]]。这里sum仅用于观察每个位置的梯度贡献。

## 2. 空洞卷积保持核参数数目

~~~python
ordinary = nn.Conv2d(1,1,3,padding=1,bias=False)
dilated = nn.Conv2d(1,1,3,padding=2,dilation=2,bias=False)
X = torch.zeros(1,1,7,7)
print(ordinary(X).shape, dilated(X).shape)
print(ordinary.weight.numel(), dilated.weight.numel())
~~~

两个输出均为7×7，核权重都为9个，但每个输出的输入覆盖跨度不同。比较数值时把两组核设成相同数值，避免把随机初始化差异混进来。

## 3. 转置卷积的最小数值例子

~~~python
values = torch.tensor([[[[1.0,2.0],[3.0,4.0]]]])
weights = torch.tensor([[[[1.0,2.0],[3.0,4.0]]]])
print(F.conv_transpose2d(values,weights,stride=1)[0,0])
print(F.conv_transpose2d(values,weights,stride=2).shape)
~~~

第一项输出3×3，第二项输出4×4。卷积核没有在这个调用前手动翻转。

## 4. 参数布局别混用

普通Conv2d的weight布局为 (Cout,Cin/groups,Kh,Kw)。
ConvTranspose2d的weight布局为 (Cin,Cout/groups,Kh,Kw)，其中Cin/Cout指该转置层自己的输入输出通道。

今天全部单通道，两者均为 (1,1,Kh,Kw)，形状差别暂时看不出来。以后扩展通道时再核对接口。

## 5. 观察output_padding的尺寸选择

~~~python
small = torch.ones(1,1,3,3)
weight = torch.ones(1,1,3,3)
first = F.conv_transpose2d(small,weight,stride=2,padding=1,output_padding=0)
second = F.conv_transpose2d(small,weight,stride=2,padding=1,output_padding=1)
print(first.shape,second.shape)
~~~

分别得到5×5与6×6。不要把output_padding直接解释成往结果右侧、下侧填零。

## 6. 手写转置卷积的贡献块

~~~text
对输入位置(i,j)：
起点 = (i*stride,j*stride)
贡献块 = image[i,j] * kernel
把贡献块加到输出对应区域，而不是覆盖原值
~~~

作业使用 += 表达重叠相加。输出先初始化为零，核不翻转，暂不添加偏置、padding或dilation。

运行 python week06/day04_conv_variants.py。示例已经提供；练习和作业的待完成部分由你填写。
