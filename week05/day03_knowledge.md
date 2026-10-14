# Week 5 / Day 3：深度学习框架与 PyTorch 基础实验

今天对应老师的[第 3 讲](https://rmcong.github.io/DL%20Course%202022/chapter3.pdf)及[实验 1](https://rmcong.github.io/DL%20Course%202022/experiment1.pdf)。目标是理解框架负责哪些工作，以及你需要自己负责哪些判断。建议用时 2 小时左右，先读第 1、3-7 节，再写练习；框架历史用于回查。

## 1. 框架到底帮我们做什么

你已经会用 NumPy 算矩阵乘法，也写过神经网络梯度。假如模型有很多层，还需自己管理每层的参数、梯度、设备和高效计算，实现会越来越繁琐。框架把这些常用能力组织起来：

| 部件 | 做什么 | 对应你写过的代码 |
|---|---|---|
| Tensor | 保存带形状、类型和设备的信息 | X.shape、X.dtype |
| 张量运算 | 加法、矩阵乘法、切片、广播 | X @ W + b |
| 自动微分 | 根据运算关系计算梯度 | loss.backward() |
| 模型与参数管理 | 管理各层和可学习参数 | nn.Module、model.parameters() |
| 优化器 | 根据梯度更新参数 | optimizer.step() |
| 数据工具 | 组织样本和批次 | Dataset、DataLoader，后续再详细练 |
| 计算后端 | 执行 CPU/GPU 上的数值运算 | 高效矩阵乘法、卷积等 |

BLAS 是基础线性代数运算库的一类接口；cuBLAS 和 cuDNN 提供针对 NVIDIA GPU 的相关计算能力。你通常通过框架调用这些底层实现，无需今天直接编写它们。

框架可以自动求导，但数据与标签是否对应、训练和验证是否混用、loss 是否适合任务，都要由你判断。

## 2. 框架历史与当前比较

课程提到 Torch、Theano、Caffe、MXNet、TensorFlow、Keras、PyTorch 等。这些名称帮助理解工具演进：从数值计算和表达式求导，到模型组件、数据管线与部署。无需按讲义里的历史热度排名选择工具。

| 名称 | 课程级认识 |
|---|---|
| Torch | 早期基于 Lua 的数值与机器学习工具 |
| Theano | 较早的符号表达式与自动微分工具 |
| Caffe | 早期常见的神经网络框架，课程介绍其配置式建模 |
| MXNet | 课程涉及的一种深度学习框架 |
| TensorFlow | 提供张量计算、自动微分、建模与训练等能力 |
| Keras | 提供较高层的模型与训练接口 |
| PyTorch | 本项目使用的张量、自动求导和建模工具 |

容易误记的一句话是“TensorFlow 只能静态图，PyTorch 只能动态图”。当前 TensorFlow 支持即时执行及图执行；PyTorch 支持即时执行及 torch.compile 等编译能力。比较框架时应说明具体版本、执行模式和任务。[TensorFlow 官方基础指南](https://www.tensorflow.org/guide/basics)、[PyTorch 编译接口](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)

即时执行的直觉是“一边运行 Python，一边执行运算”；图执行/编译可以先捕获一部分运算结构，再做优化。今天沿用 PyTorch 的普通即时执行，不增加编译练习。

## 3. 一个 Tensor 要看四件事

shape 回答“有多少样本和特征”；dtype 回答“元素如何表示”；device 回答“存在哪个计算设备”；requires_grad 回答“是否需要跟踪它的梯度”。

~~~text
X: shape=(6,2), dtype=float32, device=cpu, requires_grad=False
W: shape=(2,1), dtype=float32, device=cpu, requires_grad=True
预测 X @ W + b: shape=(6,1)
~~~

输入 X 通常不用求梯度；参数 W、b 需要求梯度。训练时仍会记录参数参与的运算，不能因为 X 不求梯度就认为整个计算没有图。

广播可能让错误形状悄悄参与计算。例如回归预测为 (6,1)，目标为 (6,) 时，逐元素减法可能广播成 (6,6)，而非你期望的六个误差。因此回归先核对预测与目标都是 (6,1)。

## 4. NumPy 与 Tensor 转换：相同数值不代表独立内存

### 4.1 两种创建方式

~~~python
array = np.array([1.0, 2.0], dtype=np.float32)
shared = torch.from_numpy(array)
copied = torch.tensor(array)
~~~

在这个普通可写 NumPy 数组的例子里，from_numpy 得到的 Tensor 与原数组共享内存；torch.tensor 则复制数据。[from_numpy 说明](https://docs.pytorch.org/docs/2.14/generated/torch.from_numpy.html)

如果随后把 array[0] 改成 9，shared[0] 也会变成 9，而 copied[0] 仍为 1。共享内存减少拷贝，但也意味着修改一方会影响另一方。今天通过一个局部小数组观察这个现象。

NumPy 的浮点数组常见默认类型是 float64，而普通 PyTorch Linear 层参数通常是 float32。转换不会自动把所有类型统一，建议读入时指定 np.float32，或显式使用 .float()。

### 4.2 Tensor 转回 NumPy

常见写法是：

~~~python
array_for_display = tensor.detach().cpu().numpy()
~~~

逐段解释：detach() 得到不跟踪梯度的 Tensor；cpu() 把数值放到 CPU；numpy() 转成 NumPy 数组。这里用于展示或分析，不能用它绕一圈再期待梯度穿过 NumPy 运算返回模型。

detach() 不等于复制内存。如果需要独立快照，例如保存参数作对照，可以再 clone()；要独立 NumPy 数组可在末尾加 .copy()。

## 5. CPU/GPU：让模型与数据在同一个地方

~~~python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Linear(2, 1).to(device)
X = X.to(device)
y = y.to(device)
~~~

to(device) 返回目标设备上的 Tensor，应接住返回值。对模型调用 .to(device) 会移动其注册的参数和缓冲区；通常先移动模型，再创建优化器。

模型在 GPU 而输入在 CPU 时，会因设备不匹配而出错。cuda.is_available() 表示当前环境是否能使用 CUDA；它为 False 不等于电脑一定没有独立显卡，可能与硬件、驱动和安装版本有关。今天默认 CPU，足够完成小实验。

## 6. 手动参数与 nn.Module：同一个模型的两种组织方式

### 6.1 手动定义参数

~~~python
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
predictions = X @ W + b
~~~

W、b 由你显式管理；计算图根据这次前向运算建立。下面的更新要放在 no_grad 中，因为“更新参数”是下一轮的准备，不需要记录为当前损失图的一部分：

~~~python
loss.backward()
with torch.no_grad():
    W -= lr * W.grad
    b -= lr * b.grad
W.grad.zero_()
b.grad.zero_()
~~~

每轮重新前向并反向；梯度默认累积，所以更新后清空。这里先得到所有梯度，再更新两组参数。W 是直接创建且要求梯度的叶子 Tensor，.grad 会保存它的梯度。

这是“手动管理参数与更新，Autograd 求导”，与 Week 3 的“手写反向传播公式”不同。两种都能加深理解，但自动化的范围不同。

### 6.2 模型类管理参数

把 nn.Linear 放到 nn.Module 的 self 属性中，框架会注册它的参数。model.parameters() 让优化器取到这些参数，免去逐个传入的工作。

~~~text
手动写法：X @ W + b，W shape=(2,1)
Linear：  X @ weight.T + bias，weight shape=(1,2)
~~~

比较参数时必须把 Linear 的 weight 转置回来。今天两种写法都从全零参数开始、使用同一数据、MSE、学习率和轮数，这样结果差异才有意义。单个线性层全零初始化可用于此对照；不要把它当作所有深层网络都适用的初始化策略。

## 7. 分类指标：准确率之外还能看什么

回归训练报告 MSE/MAE；分类报告准确率、混淆矩阵等。不要把连续回归输出直接用 argmax 算分类准确率。

二分类约定类别 1 是“正类”。混淆矩阵今天采用“行是真实类别，列是预测类别”：

~~~text
                  预测 0      预测 1
真实 0              TN          FP
真实 1              FN          TP
~~~

TN：实际是 0，预测也是 0。FP：实际是 0，错误预测为 1。FN：实际是 1，却漏判为 0。TP：实际是 1，正确预测为 1。

例如十个样本得到 TN=3、FP=1、FN=2、TP=4：

~~~text
Accuracy  = (TP+TN)/全部样本 = 7/10 = 0.7
Precision = TP/(TP+FP)       = 4/5  = 0.8
Recall    = TP/(TP+FN)       = 4/6  约为 0.667
F1        = 2*TP/(2*TP+FP+FN)= 8/11 约为 0.727
~~~

Precision 问“预测为 1 的样本中，有多少真的为 1”；Recall 问“真实为 1 的样本中，找到了多少”。F1 综合这两个方面。分母为零时必须说明约定，今天代码约定该指标记为 0。

若 100 个样本中 99 个是类别 0，全猜 0 也能得到 99% 准确率，却完全找不到正类。因此需要看类别分布和错误类型。正类、阈值以及矩阵行列方向必须写明。

## 8. 今天的学习顺序与完成标准

先运行 NumPy 共享内存与 Autograd 示例。练习 1 手动参数训练线性回归；练习 2 使用模型类完成相同任务。两者都用 SGD 式更新、lr=0.05、300 轮，更新后再计算指标。

作业使用固定分类分数练习混淆矩阵与指标，不需要额外训练分类模型。今天把任务拆开，避免把回归误差与分类准确率混在一起。

完成后，应能回答：为什么转换后的数据可能一起变？detach() 与 clone() 有何区别？手动参数是否也能自动求导？两个回归实现如何公平比较？Precision 和 Recall 分别在统计什么？

如果上一天 Logistic 的阈值或 loss 还不清楚，先回查 Day 2 第 7 节，再做分类指标作业。完成代码后发我检查。
