# 深度学习课程学习记录

[中文](README.md) | [English](README_EN.md)

本仓库记录从零学习深度学习的过程。当前以[丛润民老师《深度学习》课程](https://rmcong.github.io/proj_deep_learning_ProfessionalCourse.html)为主线，目标是掌握课程知识、完成实验，为进组做准备。仓库名称沿用早期名称，但不以实现 Transformer 作为结课标准。

学习方式是不看网课，由每日知识讲解、语法或公式速查、示例代码、小练习、当天作业与周复习组成。练习先独立完成，再检查结果和理解。

## 当前进度

| 周次 | 内容 | 状态 |
|---|---|---|
| Week 1 | Python 基础与线性模型 | 已完成 |
| Week 2 | NumPy、矩阵与梯度下降 | 已完成 |
| Week 3 | 神经网络、损失函数与反向传播 | 已完成 |
| Week 4 | PyTorch、自动求导、MLP、训练与验证 | Day 1-6 已完成；Day 7 周复习待做 |
| Week 5-12 | 老师课程其余讲次与实验 | 待学习 |

唯一有效的日程见[12 周学习计划](daily-weekly-study-plan-12-weeks.md)。计划保留前四周内容；后八周覆盖课程官网列出的 12 讲和 5 组实验。综合实验按课程要求了解五个选题，并完成其中一个。

## 已完成练习

- [Week 1](week01/)：纯 Python 线性模型、多输入与批量预测、MSE、周复习。
- [Week 2](week02/)：NumPy 数组与 shape、矩阵乘法、梯度下降、线性回归、周复习。
- [Week 3](week03/)：神经元、激活函数、损失函数、反向传播、NumPy 两层 MLP、周复习。
- [Week 4](week04/)：Tensor、Autograd、nn.Module、优化器、PyTorch MLP、XOR 训练与验证；周复习待做。

每天的资料通常包括 dayXX_knowledge.md、dayXX_pytorch_syntax.md（或对应的 Python/NumPy/公式文档）和 dayXX_*.py；复习记录在 review.md 中。

## 后续课程路线

1. 补齐课程基础、优化与正则化。
2. 卷积神经网络与 CNN 实验。
3. 循环神经网络与序列实验。
4. 无监督学习。
5. 注意力、生成模型、深度强化学习、图神经网络及课程中的行业主题。
6. 完成一个综合实验，整理可复现的报告与进组准备材料。

## 运行环境

建议使用 Python 虚拟环境；依赖版本见 [requirements.txt](requirements.txt)。仓库中的练习默认以 CPU 为主。

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python week04/day06_train_validate.py
~~~

macOS/Linux 激活命令为 source .venv/bin/activate。运行某一天的练习前，可先阅读同目录的知识讲解与语法文档。

## 课程完成标准

能解释主要模型的结构与训练目标，独立运行并分析实验，清楚区分训练与验证数据，并提交一个可复现的综合实验报告。进组后的研究方向以老师安排为准。
