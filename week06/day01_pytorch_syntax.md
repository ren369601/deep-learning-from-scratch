# Week 6 / Day 1：图像、窗口与二维互相关写法

## 1. 创建二维输入与增加维度

~~~python
import torch
from torch.nn import functional as F

image = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
batch = image.unsqueeze(0).unsqueeze(0)
print(image.shape, batch.shape)  # (3,3) 与 (1,1,3,3)
restored = batch[0, 0]
print(torch.equal(image, restored))
~~~

unsqueeze(0) 在开头增加一个长度为 1 的维度，不改变像素值。本周使用 (N,C,H,W) 顺序。

## 2. 窗口切片与逐元素乘加

~~~python
kernel = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
window = image[0:2, 0:2]
value = (window * kernel).sum()
print(window, value.item())
~~~

切片右端不包含在结果内，因此 0:2 取位置 0、1。* 对应位置相乘，sum 汇总；@ 不是今天窗口运算的写法。

## 3. 手动零填充

~~~python
padding = 1
height, width = image.shape
padded = image.new_zeros((height + 2*padding, width + 2*padding))
padded[padding:padding+height, padding:padding+width] = image
print(padded)
~~~

new_zeros 沿用 image 的 dtype 和 device。中间写入原图，四周保留 0。padding=0 时同样能使用这个切片形式。

## 4. 输出尺寸中的整除

~~~python
height, kernel_height, stride, padding = 6, 3, 2, 1
output_height = (height + 2*padding - kernel_height) // stride + 1
print(output_height)
~~~

// 是向下整除；计算前检查 stride>0、padding>=0，且填充后输入不小于核。宽度按自己的核宽单独计算。

## 5. 仅用于验证的库结果

~~~python
reference = F.conv2d(image[None, None], kernel[None, None], stride=1, padding=0)
print(reference.shape)
print(reference[0, 0])
~~~

None 索引在这里与两次 unsqueeze 的效果相同。reference 是四维；手写结果是二维，先取 [0,0] 再比较。

## 6. 核翻转与浮点比较

~~~python
flipped = torch.flip(kernel, dims=(0, 1))
print(flipped)
expected = torch.full((2, 2), -4.0)
assert torch.allclose(reference[0, 0], expected)
~~~

flip 反转行列，T 只是转置。allclose 用容差比较浮点数，适合检查不同计算实现。

## 7. 今日运行方式

在虚拟环境中运行 python week06/day01_convolution.py。示例已完成，输出尺寸函数、手写互相关及练习调用留给你补齐。只有作业内调用 verify_manual_implementation 后，才会运行提供的数值核对。
