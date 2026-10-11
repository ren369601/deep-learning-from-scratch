# Week 4 / Day 1：Tensor 练习用到的 PyTorch 写法

这份是今天的语法速查。先读 `day01_knowledge.md` 理解概念，再用本页查写法，最后完成 `day01_tensor.py` 中的 TODO。

## 1. 导入与创建 Tensor

```python
import torch

X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
targets = torch.tensor([0, 1], dtype=torch.long)
zeros = torch.zeros(2, 3)
numbers = torch.arange(6, dtype=torch.float32)
```

`torch.float32` 适合今天的输入与权重，`torch.long` 是常见的整数分类标签类型。`torch.arange(6)` 依次产生 `0` 到 `5`。

## 2. 查看 shape、类型与设备

```python
print(X.shape)   # torch.Size([2, 2])
print(X.ndim)    # 2
print(X.dtype)   # torch.float32
print(X.device)  # cpu
```

二维 Tensor 的 shape 通常可按 `(样本数, 特征数)` 理解。`X.shape == (2, 2)` 的比较结果为 `True`。

## 3. 逐元素运算、矩阵乘法与广播

```python
X * 2               # 每个元素乘 2
X + 1               # 每个元素加 1
X @ W               # 矩阵乘法；检查内侧维度
X @ W + b           # b 的 shape 为 (输出特征数,)，广播到每一行
```

例如 `(3, 2) @ (2, 4) -> (3, 4)`。`*` 不等于 `@`。

## 4. 改变 shape 与转置


```python
matrix = numbers.reshape(2, 3)  # (2, 3)
transposed = matrix.T            # (3, 2)，二维时交换行列
```

`reshape` 按目标形状重新组织元素，`.T` 交换行列。不能只凭输出 shape 判断两者得到的值相同。

## 5. 沿指定维度求平均

```python
column_means = matrix.mean(dim=0)                 # (3,)
row_means = matrix.mean(dim=1)                    # (2,)
column_means_row = matrix.mean(dim=0, keepdim=True)  # (1, 3)
centered = matrix - column_means_row             # (2, 3)
```

`dim=0` 沿行方向合并，得到每列的平均；`keepdim=True` 保留这一维，方便观察广播。

## 6. 打印与快速检查

```python
print("shape:", matrix.shape)
print("values:\n", matrix)
assert matrix.shape == (2, 3)
```

做练习时，每写完一个表达式就先打印 shape，再对照手算值。今天只需 CPU Tensor，不需要 `requires_grad`；自动求导留到 Day 2。
