# 从零实现 Transformer

[中文](README.md) | [English](README_EN.md)

这是一个从 Python 和数学基础出发，逐步学习深度学习并最终从零实现 Transformer 的长期学习项目。

项目不以“调用现成 Transformer API”为终点，而是希望真正理解模型中的每一个核心部分：矩阵计算、梯度下降、反向传播、Attention、Multi-Head Attention、Encoder、Decoder，以及完整模型的训练与推理。

## 项目目标

完成本项目后，希望具备以下能力：

- 理解神经网络训练的基本原理。
- 熟练使用 NumPy 和 PyTorch 处理张量与矩阵。
- 能够独立实现常见神经网络模块。
- 理解并手写 Attention 和 Multi-Head Attention。
- 从零实现 Transformer Encoder、Decoder 和完整模型。
- 阅读并复现《Attention Is All You Need》中的核心结构。
- 能够阅读深度学习论文与 GitHub 项目，并独立调试模型。

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| Week 1 | Python 基础与线性模型 | 已完成 |
| Week 2 | NumPy、矩阵计算与梯度下降 | 已完成 |
| Week 3 | 神经元、激活函数与反向传播 | 下一阶段 |
| Week 4 | PyTorch 基础与 MLP | 未开始 |
| Week 5 | 训练流程与 CNN | 未开始 |
| Week 6 | RNN、LSTM 与序列建模 | 未开始 |
| Week 7 | Attention | 未开始 |
| Week 8 | Multi-Head Attention 与 Mask | 未开始 |
| Week 9 | Transformer Encoder | 未开始 |
| Week 10 | Transformer Decoder 与完整模型 | 未开始 |
| Week 11 | 论文阅读与复现 | 未开始 |
| Week 12 | 训练、实验与项目整理 | 未开始 |

详细安排见 [12 周学习计划](daily-weekly-study-plan-12-weeks.md)。仓库中也保留了更舒缓的 [24 周学习计划](deep-learning-24-week-plan.md)。

## 已完成内容

### Week 1：Python 基础与线性模型

- `y = wx + b` 与多特征线性模型。
- Python 函数、list、循环和 class。
- 样本、batch 与批量预测。
- 使用 class 封装线性模型。
- 手写 MSE，并枚举候选参数。
- 完成纯 Python 线性回归雏形。

复习资料：[Week 1 Review](week01/review.md)

### Week 2：NumPy 与梯度下降

- NumPy array、shape、ndim 和 dtype。
- reshape、transpose 和 broadcasting。
- 点积与矩阵乘法。
- NumPy 向量化 MSE。
- 梯度下降、学习率和训练循环。
- 使用 `X.T @ errors` 训练多特征线性回归。

复习资料：[Week 2 Review](week02/review.md)

## 学习路线

```text
Python / NumPy
        ↓
数学基础与梯度下降
        ↓
神经网络与反向传播
        ↓
PyTorch 与 MLP
        ↓
CNN / RNN / LSTM
        ↓
Attention
        ↓
Multi-Head Attention 与 Mask
        ↓
Transformer Encoder / Decoder
        ↓
论文阅读、复现与完整项目
```

## 项目结构

```text
.
├── README.md
├── README_EN.md
├── requirements.txt
├── daily-weekly-study-plan-12-weeks.md
├── deep-learning-24-week-plan.md
├── week01/
│   ├── dayXX_knowledge.md
│   ├── dayXX_python_syntax.md
│   ├── dayXX_*.py
│   ├── notes.md
│   └── review.md
└── week02/
    ├── dayXX_knowledge.md
    ├── dayXX_numpy_syntax.md
    ├── dayXX_*.py
    ├── linear_regression_numpy.py
    ├── notes.md
    └── review.md
```

后续每一周都会延续相同结构：

```text
知识讲解 -> 语法与公式 -> 示例代码 -> 小练习 -> 当天作业 -> 周复习
```

## 环境配置

克隆仓库：

```bash
git clone https://github.com/ren369601/deep-learning-from-scratch.git
cd deep-learning-from-scratch
```

创建虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 运行示例

运行 Week 1 线性回归综合练习：

```bash
python week01/day06_week01_review.py
```

运行 Week 2 NumPy 多特征线性回归：

```bash
python week02/linear_regression_numpy.py
```

运行 Week 2 综合复习：

```bash
python week02/day07_week02_review.py
```

## 学习原则

- 理论和代码同步学习，不把二者分开。
- 每次矩阵计算前先分析 shape。
- 尽量先独立完成练习，再对照讲解和示例。
- 不只追求代码能运行，还要能解释每一步的作用。
- 每周整理笔记、运行全部代码并提交 GitHub。

## 最终成果

项目最终会形成一个不依赖现成 Transformer 模型 API 的完整实现，包括：

- Scaled Dot-Product Attention
- Multi-Head Attention
- Padding Mask 与 Causal Mask
- Positional Encoding
- Feed Forward Network
- Layer Normalization 与 Residual Connection
- Transformer Encoder 与 Decoder
- 训练循环与推理生成
- 《Attention Is All You Need》阅读与复现笔记

当前项目仍在持续学习和完善中。
