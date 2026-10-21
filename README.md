# 深度学习基础与实验

[中文](README.md) | [English](README_EN.md)

基于 Python、NumPy 和 PyTorch 的深度学习学习与实验记录，沿[丛润民老师《深度学习》课程](https://rmcong.github.io/proj_deep_learning_ProfessionalCourse.html)逐步推进。重点展示基础算法实现、模型训练与验证，以及控制变量的实验分析；目前为课程实践项目，非原创研究成果。

## 已完成内容

- **基础实现（Week 1–3）**：线性回归、矩阵运算、梯度下降、反向传播与 NumPy 两层 MLP。
- **PyTorch 实践（Week 4–5）**：自动求导、分类训练、训练／验证划分、K 折验证、优化器与正则化对照。
- **卷积网络（Week 6 Day 1–6）**：手写二维互相关、多通道卷积与转置卷积；CNN 分类；LeNet/Inception 教学实现；残差块及普通 CNN／残差网络配对实验。

## 代表实验

| 实现 | 可查看内容 |
|---|---|
| [NumPy MLP](week03/day06_mlp_train.py) | 前向传播、梯度计算与参数更新 |
| [手写转置卷积](week06/day04_conv_variants.py) | 贡献块累加、三组 PyTorch 数值对照 |
| [残差网络对照](week06/day06_resnet_lab.py) | 三个初始化种子，相同参数量、初始权重和批次顺序，仅切换残差相加 |

残差对照采用本地合成横线／竖线数据。两组网络均为 3658 个参数，最终验证准确率均为 100%；本次残差网络最终验证损失更低。[配置、结果与局限](week06/day06_knowledge.md)均已记录；该结果不代表自然图像性能或论文级复现。

## 运行

依赖版本见 [requirements.txt](requirements.txt)，默认使用 CPU。创建并激活虚拟环境后，在仓库根目录运行：

```powershell
python -m pip install -r requirements.txt
python -X utf8 week06/day06_resnet_lab.py
```

每个学习日包含知识讲解、语法速查与练习代码。当前进度至 Week 6 Day 6，后续内容见 [12 周计划](daily-weekly-study-plan-12-weeks.md)；尚未完成全部课程。讲解与部分参考解答使用 AI 辅助编写，具体补全与运行记录保留在学习材料中，不将助手补全等同于独立实现。
