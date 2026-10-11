# Week 4 / Day 1：PyTorch Tensor、shape 与基础运算

本周开始把上周用 NumPy 写的神经网络迁移到 PyTorch。今天只学 Tensor，不急着用自动求导；Day 2 再学习 Autograd。

## 1. Tensor 是什么

Tensor 可以先理解为 PyTorch 的多维数组。和 NumPy 数组一样，它有 shape、数据类型，可以进行逐元素运算、矩阵乘法、转置和聚合。PyTorch 还能够记录求导所需的计算，并可以把 Tensor 放在 GPU 上；今天先用 CPU。

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
print(x.shape)   # torch.Size([2, 2])，可与 (2, 2) 比较
print(x.ndim)    # 2
print(x.dtype)   # torch.float32
print(x.device)  # cpu
```

训练数据一般用浮点数，分类标签通常用整数：

```python
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
targets = torch.tensor([0, 1], dtype=torch.long)
```

`targets` 的 shape 是 `(2,)`，而不是 `(2, 1)`。标签类型的具体要求与所用损失函数有关，之后学习分类损失时再细讲。

## 2. 与 NumPy 的对应关系

| NumPy | PyTorch | 用途 |
|---|---|---|
| `np.array(data)` | `torch.tensor(data)` | 从数据创建数组/Tensor |
| `np.zeros((2, 3))` | `torch.zeros(2, 3)` | 全零 |
| `x.shape` | `x.shape` | 查看维度 |
| `x.reshape(3, 2)` | `x.reshape(3, 2)` | 重排 shape |
| `x.T` | `x.T`（二维） | 转置 |
| `x @ w` | `x @ w` | 矩阵乘法 |
| `x.mean(axis=0)` | `x.mean(dim=0)` | 沿某一维求平均 |

注意：`torch.tensor(data)` 会创建新的 Tensor。今天不必纠结 Tensor 与 NumPy 共享内存、`view` 等细节。

## 3. Shape 与矩阵乘法

沿用两层 MLP 的记法：`X (N, D)`、`W (D, H)`、`b (H,)`。

```python
Z = X @ W + b
```

这里 `(N, D) @ (D, H) -> (N, H)`，偏置 `(H,)` 被广播到每个样本。`*` 是逐元素乘法；`@` 才是矩阵乘法，不要混用。

举例：

```python
X = torch.tensor([[1., 2.], [3., 4.]])   # (2, 2)
W = torch.tensor([[1., 0.], [0., 2.]])   # (2, 2)
b = torch.tensor([10., 20.])            # (2,)
Z = X @ W + b                            # [[11., 24.], [13., 28.]]
```

## 4. 逐元素运算与广播

```python
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])  # (2, 3)
bias = torch.tensor([10., 20., 30.])          # (3,)
print(x + bias)                                # (2, 3)
print(x * 2)                                   # (2, 3)
```

PyTorch 的广播和 NumPy 思路相同：从右侧维度开始比较，维度相同或其中一个为 1 时可扩展。广播很方便，也可能掩盖意外的 shape；训练时要经常打印或断言 shape。

## 5. `reshape`、转置、聚合

```python
numbers = torch.arange(6, dtype=torch.float32)
matrix = numbers.reshape(2, 3)  # [[0, 1, 2], [3, 4, 5]]
transposed = matrix.T            # (3, 2)，交换行列

column_means = matrix.mean(dim=0)                    # (3,)
column_means_row = matrix.mean(dim=0, keepdim=True)  # (1, 3)
centered = matrix - column_means_row                 # (2, 3)
```

`reshape` 改变元素组织方式；二维 `.T` 交换两个轴。二者即使偶尔得到一样的 shape，数值排列也通常不同。`keepdim=True` 保留被聚合的轴，在检查广播过程时尤其直观。

## 6. 今天的易错点

1. `torch.tensor([1, 2])` 默认可能是整数类型；需要浮点训练数据时显式写 `dtype=torch.float32`。
2. `x * w` 与 `x @ w` 不同：前者逐元素，后者矩阵乘法。
3. `mean(dim=0)` 沿样本轴聚合，结果通常是每列的平均；`mean(dim=1)` 是每行平均。
4. `reshape` 不等于转置；先看 shape，再看元素是否按预期排列。
5. 先把 CPU 上的基础运算和 shape 弄清楚，暂不需要 CUDA 或 `requires_grad`。

## 7. 今日完成标准

- 能创建浮点 Tensor 和整数标签 Tensor，读懂 `shape`、`dtype`、`device`。
- 能预测 `(N, D) @ (D, H) + (H,)` 的输出 shape。
- 能解释 `*`、`@`、`reshape`、`.T`、`mean(dim=...)` 的区别。
- 完成 `day01_tensor.py` 中的两个练习和当天作业，写下每个结果的 shape。

安装与验证：项目虚拟环境使用 Python 3.14。按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择 Windows、Pip、CPU 后安装；本周只用 `torch`，无需额外安装 `torchvision` 或 `torchaudio`。在项目根目录用 `.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"` 验证。

明天进入 Autograd，比较手写梯度与 `backward()` 算出的梯度。
