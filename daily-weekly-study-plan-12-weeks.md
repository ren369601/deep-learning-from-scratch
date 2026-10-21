# 深度学习课程：12 周学习计划（Week 5-12 修订）

Week 1-4 已完成；下面的 Week 1-4 表格原样保留。Week 5 起以[丛润民老师《深度学习》课程](https://rmcong.github.io/proj_deep_learning_ProfessionalCourse.html)为主线，覆盖官网列出的 12 讲和 5 组实验；不用看网课，改为每天由我讲解、演示，再由你动手练习。当前目标是学扎实课程内容、完成课程实验，为进组做准备；不以实现某一种模型作为额外验收标准。

当前进度：Week 5 Day 1-7 的代码与周复习记录已完成，Day 5-7 含参考补全实现；Day 1 个人应用分析待补。Week 6 Day 1-5 代码已完成并运行核对，其中部分由助手按要求补全，不等于独立掌握。Week 6 Day 6（ResNet）三份材料、练习和作业已补齐，运行结果见当天知识讲解；Day 7 周复习待进行。

执行约定：

- 每天仍交付三份材料：知识讲解 `.md`、Python/PyTorch 语法或公式速查 `.md`、练习 `.py`（概念日可用练习 `.md`）。代码格式沿用 Week 3：示例、小练习、当天作业、分段运行入口。
- 每天建议 1.5-2.5 小时；Day 7 用于复习、补进度和整理 GitHub。新内容难度较高时，以能讲清原理、跑通小实验为准，不强求一天做完大模型。
- “覆盖所有内容”指课程的**全部讲次、主要知识点及五组实验的要求和选择**都进入学习与验收；实验 5 按原要求从五个选题中完成一个，不等于五个课题都做完。
- 课程资料发布于 2022 年。框架现状和“前沿趋势”会区分当年的教学内容与学习当天的现状；第 12 讲官网只提供行业讲座视频，我们以官网主题和公开文字资料学习，不声称复述视频细节。
- 默认用 CPU 可运行的小数据。GPU、较大数据集和复杂复现作为条件允许时的扩展。

## 12 周总览

| 周数 | 阶段 | 本周主题 | 本周项目产出 |
|---|---|---|---|
| Week 1 | Python 基础 | Python、函数、class、线性模型 | 单输入 / 多输入线性模型 |
| Week 2 | NumPy + 数学基础 | array、shape、矩阵乘法、梯度直觉 | NumPy Linear Regression |
| Week 3 | 神经网络基础 | Neuron、Activation、Loss、Backpropagation | NumPy 两层 MLP |
| Week 4 | PyTorch 基础 | Tensor、Autograd、nn.Module、Optimizer | PyTorch MLP |
| Week 5 | 课程基础补齐与优化 | 第 1-5 讲、实验 1-2 的未覆盖点 | 分类基线、MLP 训练与正则化对照 |
| Week 6 | 卷积神经网络 | 第 6-7 讲、实验 3 | CNN 分类与 ResNet/空洞卷积小实验 |
| Week 7 | 循环神经网络 | 第 8-9 讲、实验 4 | RNN/LSTM/GRU 时序任务对照 |
| Week 8 | 无监督学习 | 第 10 讲 | Hopfield/RBM 概念图与自编码器实验 |
| Week 9 | 深度学习前沿 | 第 11 讲 | Attention、VAE/GAN、深度强化学习、GNN 入门实验与笔记 |
| Week 10 | 行业应用与选题 | 第 12 讲、实验 5 选题 | 五个选题比较、选定一个、建立可复现基线 |
| Week 11 | 综合实验 | 实验 5 实施 | 选题模型、训练、评估与对照 |
| Week 12 | 全课程复盘 | 第 1-12 讲、实验 1-5 验收 | 实验报告、知识索引与 GitHub 项目整理 |

## Week 1：Python 基础与线性模型

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 线性模型 `y = wx + b`，变量、函数、list、for、class | 单输入线性模型 | `week01/day01_python_linear_model.py` |
| Day 2 | 多输入线性模型，feature 和 weight 的对应关系 | `y = w1*x1 + w2*x2 + b` | `week01/day02_multi_feature_linear.py` |
| Day 3 | 嵌套 list、批量样本、批量预测 | 多个样本批量预测 | `week01/day03_batch_prediction.py` |
| Day 4 | 函数拆分、class 加强、代码结构 | 完整 `LinearModel` 类 | `week01/day04_linear_model_class.py` |
| Day 5 | 小项目：不用 NumPy 写 Linear Regression 雏形 | 手动预测、计算误差 | `week01/day05_linear_regression_plain_python.py` |
| Day 6 | 复习与整理 | 重写核心代码，更新笔记 | `week01/review.md` |
| Day 7 | 休息 / 补进度 / GitHub 提交 | 整理文件并提交 | GitHub commit |

## Week 2：NumPy、矩阵和梯度直觉

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | NumPy array、shape、dtype | 创建向量和矩阵 | `week02/day01_numpy_array.py` |
| Day 2 | reshape、transpose、广播 | shape 变化练习 | `week02/day02_shape_ops.py` |
| Day 3 | dot product、矩阵乘法 | 手写 dot，对比 NumPy | `week02/day03_dot_matmul.py` |
| Day 4 | MSE Loss、误差直觉 | 计算预测误差 | `week02/day04_mse_loss.py` |
| Day 5 | 梯度下降直觉 | 手动更新 `w` 和 `b` | `week02/day05_gradient_descent.py` |
| Day 6 | NumPy Linear Regression | 完整训练线性回归 | `week02/linear_regression_numpy.py` |
| Day 7 | 复习 | 整理 shape、loss、gradient | `week02/review.md` |

## Week 3：神经网络与反向传播

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Neuron、Linear Layer | 写一个单神经元和线性层 | `week03/day01_neuron_linear.py` |
| Day 2 | Activation Function | Sigmoid、Tanh、ReLU、Softmax | `week03/day02_activation.py` |
| Day 3 | Loss Function | MSE、Cross Entropy 直觉 | `week03/day03_loss.py` |
| Day 4 | Backpropagation 直觉 | 手算简单梯度 | `week03/day04_backprop_intro.md` |
| Day 5 | NumPy MLP forward | 两层 MLP 前向传播 | `week03/day05_mlp_forward.py` |
| Day 6 | NumPy MLP train | 简化版反向传播和训练 | `week03/day06_mlp_train.py` |
| Day 7 | 复习 | 整理 forward / backward | `week03/review.md` |

## Week 4：PyTorch 基础与 MLP

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Tensor、shape、基础运算 | Tensor 操作练习 | `week04/day01_tensor.py` |
| Day 2 | Autograd | `requires_grad`、`backward()` | `week04/day02_autograd.py` |
| Day 3 | nn.Module | 自定义模型类 | `week04/day03_nn_module.py` |
| Day 4 | Optimizer | SGD、Adam 基本使用 | `week04/day04_optimizer.py` |
| Day 5 | PyTorch MLP | 搭建两层 MLP | `week04/day05_pytorch_mlp.py` |
| Day 6 | 训练与验证 | 完成标准训练循环 | `week04/day06_train_validate.py` |
| Day 7 | 复习 | 整理 PyTorch 标准模板 | `week04/review.md` |

## Week 5：补齐课程基础、优化与正则化

前四周的练习不返工；这里只补课程第 1-4 讲和实验 1-2 中尚未系统覆盖的知识，再学习第 5 讲。

| 天数 | 学习内容 | 当天练习 | 主要产出 |
|---|---|---|---|
| Day 1 | 第 1 讲：AI/机器学习/深度学习关系、发展与应用；学习任务的基本元素 | 用已有线性模型解释数据、模型、损失、训练和泛化 | week05/day01_course_map.py |
| Day 2 | 第 2 讲：线性代数、微积分、概率统计回顾；线性/逻辑/Softmax 回归与感知机 | 手算并实现一个二分类基线，比较回归与分类输出 | week05/day02_classification_baselines.py |
| Day 3 | 第 3 讲与实验 1：框架组件、TensorFlow/PyTorch；Tensor 与 NumPy 转换、设备、自动求导、线性回归两种写法、分类指标 | 用手动参数和 nn.Module 各训练一次，报告准确率/混淆矩阵 | week05/day03_framework_lab.py |
| Day 4 | 第 4 讲与实验 2：XOR、前馈网络、前反向传播、手写/框架 MLP；模型选择与 K 折验证 | 在 XOR 或小型表格任务上做 MLP 与 K 折比较 | week05/day04_mlp_model_selection.py |
| Day 5 | 第 5 讲：mini-batch、SGD/动量/Adam、学习率、参数初始化、输入预处理 | 固定数据和随机种子，比较优化器及学习率 | week05/day05_optimization.py |
| Day 6 | 第 5 讲与实验 2：归一化、超参数、欠/过拟合、权重衰减、Dropout、早停 | 造一个易过拟合的小任务，记录训练/验证曲线与正则化对照 | week05/day06_regularization.py |
| Day 7 | 第 1-5 讲、实验 1-2 回顾 | 补齐运行结果、错误分析和课程知识图 | week05/review.md |

## Week 6：卷积神经网络（第 6-7 讲、实验 3）

| 天数 | 学习内容 | 当天练习 | 主要产出 |
|---|---|---|---|
| Day 1 | 图像张量、卷积/互相关、卷积核、步长、填充和输出尺寸 | 不用现成卷积层手写二维卷积，对照尺寸计算 | week06/day01_convolution.py |
| Day 2 | 局部连接、参数共享、平移等变性；多通道卷积与池化 | 比较参数量，观察平移/池化前后的输出 | week06/day02_cnn_principles.py |
| Day 3 | CNN 层级结构与图像分类训练 | 用小数据训练 CNN，记录训练/验证指标 | week06/day03_cnn_classifier.py |
| Day 4 | 卷积核梯度、转置卷积、空洞卷积 | 跟踪 shape；写空洞或转置卷积小例子 | week06/day04_conv_variants.py |
| Day 5 | LeNet、AlexNet、Inception：结构、历史动机与代价 | 画数据流，计算关键层参数量，搭建小型分支分类器 | week06/day05_cnn_architectures.py |
| Day 6 | ResNet 残差连接；实验 3 综合练习 | 实现残差块，比较普通 CNN 与小型 ResNet | week06/day06_resnet_lab.py |
| Day 7 | CNN 复习与实验报告 | 检查手写卷积、空洞卷积、分类和 ResNet 的结果 | week06/review.md |

## Week 7：循环神经网络（第 8-9 讲、实验 4）

| 天数 | 学习内容 | 当天练习 | 主要产出 |
|---|---|---|---|
| Day 1 | 序列数据、时间窗、训练/测试切分与泄漏；变长序列填充 | 先按时间切分，再生成窗口和 batch | week07/day01_sequence_data.py |
| Day 2 | 循环状态与展开、基础 RNN、时间反向传播 BPTT | 手写 RNN cell 前向并检查 shape | week07/day02_rnn_bptt.py |
| Day 3 | 双向 RNN、深层 RNN 与序列输出方式 | 对比单向/双向输出及应用限制 | week07/day03_bidirectional_deep.py |
| Day 4 | 长期依赖、梯度消失/爆炸；LSTM 的门与记忆单元 | 手写 LSTM 单步前向，解释各门作用 | week07/day04_lstm.py |
| Day 5 | GRU 的更新/重置门；RNN/LSTM/GRU 比较 | 在相同输入上运行三种单元并计数参数 | week07/day05_gru_compare.py |
| Day 6 | 实验 4：时序预测或序列分类；变长序列的 mask/pack | 同一任务训练 RNN、LSTM、GRU，固定切分和指标 | week07/day06_rnn_lab.py |
| Day 7 | RNN 复习与实验报告 | 总结 BPTT、长程依赖和三种模型的结果 | week07/review.md |

## Week 8：无监督学习（第 10 讲）

这一周的能量模型以理解结构和训练思想为主；不要求在有限设备上完整训练 DBM/DBN。

| 天数 | 学习内容 | 当天练习 | 主要产出 |
|---|---|---|---|
| Day 1 | 无监督学习的目标；Hopfield 网络、能量与记忆检索 | 做小型二值模式恢复实验 | week08/day01_hopfield.py |
| Day 2 | Boltzmann 机、受限 Boltzmann 机（RBM）及对比散度思想 | 画可见/隐藏单元，跟踪一次采样更新 | week08/day02_boltzmann_rbm.md |
| Day 3 | 深度 Boltzmann 机（DBM）、深度信念网络（DBN）、逐层预训练 | 比较两类结构与训练难点 | week08/day03_deep_energy_models.md |
| Day 4 | 自编码器（AE）的编码、解码、重构损失和瓶颈 | 训练小型 AE 并观察重构误差 | week08/day04_autoencoder.py |
| Day 5 | 去噪 AE、稀疏 AE 及各自的约束 | 对比有噪/无噪输入，观察重构 | week08/day05_ae_variants.py |
| Day 6 | 堆叠 AE 与逐层预训练；无监督表征的评价 | 对比 AE 表征与原始输入的简单下游任务 | week08/day06_stacked_ae.py |
| Day 7 | 无监督学习总复习 | 解释 Hopfield、RBM/DBM/DBN 与 AE 家族的区别 | week08/review.md |

## Week 9：深度学习前沿（第 11 讲）

讲义中的“前沿”按 2022 年的语境阅读；学习当天再核对这些方向目前的状况。Attention 作为第 11 讲的课程内容学习，不额外扩展为独立项目要求。

| 天数 | 学习内容 | 当天练习 | 主要产出 |
|---|---|---|---|
| Day 1 | 注意力机制：查询、键、值与加权汇聚；自注意力入门 | 手算并实现单头缩放点积注意力 | week09/day01_attention.py |
| Day 2 | 深度生成模型与变分自编码器（VAE）：隐变量、重参数化、重构与 KL | 在小数据上运行最简 VAE 或核对损失各项 | week09/day02_vae.py |
| Day 3 | 生成对抗网络（GAN）：生成器、判别器、对抗目标与训练不稳定 | 用一维玩具分布做最小 GAN 实验 | week09/day03_gan.py |
| Day 4 | 深度强化学习：状态、动作、奖励、价值函数、DQN 思想 | 在极小环境中列出转移并计算一步目标值 | week09/day04_deep_rl.md |
| Day 5 | 图神经网络：图数据、邻域聚合、消息传递与图卷积 | 手算并编码一层邻居聚合 | week09/day05_gnn.py |
| Day 6 | 深度学习局限与发展方向；比较 Attention/VAE/GAN/RL/GNN 的任务边界 | 写一页案例分析，区分课程当年观点和当前事实 | week09/day06_frontiers_review.md |
| Day 7 | 第 11 讲复习 | 能解释五类方法的目标、输入输出和主要限制 | week09/review.md |

## Week 10：行业应用与综合实验选题（第 12 讲、实验 5）

第 12 讲官网为行业讲座，没有配套 PDF。当天以官网列出的“AI 框架发展趋势与 MindSpore 实践”主题和公开文字资料讲解，不把它说成讲座逐字内容。实验 5 原要求五选一。

| 天数 | 学习内容 | 当天练习 | 主要产出 |
|---|---|---|---|
| Day 1 | AI 框架的组件与发展、MindSpore 实践主题；与 PyTorch 概念对照 | 写同一训练任务的框架组件对照表 | week10/day01_framework_industry.md |
| Day 2 | 实验 5 五个方向：出租车流量预测、微博情感分类、单目标跟踪、顶会论文复现、OneFlow 竞赛 | 比较数据、算力、指标、许可与完成风险 | week10/day02_capstone_options.md |
| Day 3 | 从五个方向选一个；任务定义与可用数据 | 写任务、数据来源、许可、输入输出和可验收目标 | week10/day03_project_scope.md |
| Day 4 | 数据检查、训练/验证/测试划分、数据泄漏 | 实现可复现的读取和切分 | week10/day04_data_pipeline.py |
| Day 5 | 任务指标与简单基线 | 先跑一个朴素基线并保存结果 | week10/day05_baseline.py |
| Day 6 | 模型方案、资源预算和实验表 | 设计模型与最小可行实验；跑通一个 batch | week10/day06_project_design.md |
| Day 7 | 选题阶段复盘 | 检查基线、数据许可、资源和下一周风险 | week10/review.md |

## Week 11：综合实验实施（实验 5）

所有结果必须在相同数据切分和指标下比较。若选“顶会论文复现”，先写清楚是核心机制的小规模复现，不能把玩具实验冒充论文原始结果。

| 天数 | 学习内容 | 当天练习 | 主要产出 |
|---|---|---|---|
| Day 1 | 针对选题实现最小模型 | 从输入到预测完成一次前向与损失计算 | week11/day01_model.py |
| Day 2 | 训练循环、日志与保存 | 跑通训练/验证并保存配置和权重 | week11/day02_train.py |
| Day 3 | 调试与训练稳定性 | 单 batch 过拟合、检查梯度和数据标签 | week11/day03_debug.md |
| Day 4 | 与 Week 10 基线比较 | 固定评价脚本，记录指标和样例 | week11/day04_evaluate.py |
| Day 5 | 单变量对照/消融 | 改一个组件或超参数，记录公平比较 | week11/day05_ablation.md |
| Day 6 | 错误分析与局限 | 按样例分类错误，说明数据/算力限制 | week11/day06_error_analysis.md |
| Day 7 | 综合实验阶段报告 | 整理任务、数据、方法、结果、问题 | week11/review.md |

## Week 12：全课程复盘与项目整理

| 天数 | 学习内容 | 当天练习 | 主要产出 |
|---|---|---|---|
| Day 1 | 第 1-5 讲、实验 1-2 总复习 | 口述从线性回归到 MLP 的训练全过程 | week12/day01_foundations_review.md |
| Day 2 | 第 6-9 讲、实验 3-4 总复习 | 解释 CNN 与 RNN/LSTM/GRU 的结构、shape 和适用任务 | week12/day02_cnn_rnn_review.md |
| Day 3 | 第 10-12 讲总复习 | 对比能量模型/AE、生成模型、Attention、RL、GNN、行业框架 | week12/day03_advanced_review.md |
| Day 4 | 实验 5 重跑与结果核查 | 从固定种子和配置重跑关键结果 | week12/day04_reproduce.md |
| Day 5 | 实验报告定稿 | 写方法、对照、结果、局限和引用 | week12/day05_final_report.md |
| Day 6 | GitHub 项目整理 | 补 README、运行步骤、数据许可、目录索引和测试 | README.md 与项目目录 |
| Day 7 | 课程验收与进组准备 | 查缺补漏，整理课程知识、实验成果和待向老师请教的问题 | week12/review.md |

## 每天学习模板

每天仍按“知识讲解 → 示例代码 → 小练习 → 当天作业 → GitHub 项目积累”的顺序。计划表中的“主要产出”是练习/报告文件；此外当天还会在同一 week 文件夹下配套 dayXX_knowledge.md 和 dayXX_pytorch_syntax.md（数学日用 dayXX_formula.md）。复习日以 review.md 为主。

知识讲解采用详细的小步推导：新术语出现时先解释含义；尽量沿用同一组小数字，完整展开每一步计算；先数位置、观察结果，再推导公式；代码逐句对应含义与 shape；安排可立即核对的小问题，并列出常见误解。内容较多时标明分次阅读的顺序，避免只增加篇幅而省略中间步骤。

代码练习沿用 Week 3 及以前的文件风格：先示例，后两道左右的小练习和当天作业，保留分段运行入口。当天作业由你写，我检查后再调整下一天难度。

从 Week 6 Day 3 起，新生成的 Python 文件使用中文注释、中文文档字符串、中文练习说明与输出提示；函数名、变量名和库接口保留惯用英文，便于阅读和检索。

## 课程覆盖核对

| 官网内容 | 本计划位置 | 验收重点 |
|---|---|---|
| 第 1 讲：深度学习介绍 | Week 5 Day 1 | 术语、发展与应用 |
| 第 2 讲：数学与机器学习基础 | Week 1-3；Week 5 Day 2 | 回归、分类、感知机及基本数学 |
| 第 3 讲：框架与 PyTorch | Week 4；Week 5 Day 3 | Tensor、自动求导、框架组件、实验 1 |
| 第 4 讲：深度前馈网络 | Week 3-4；Week 5 Day 4 | XOR、前/反向传播、MLP |
| 第 5 讲：优化与正则化 | Week 5 Day 5-6 | 优化器、初始化、归一化、过拟合 |
| 第 6-7 讲：CNN I/II | Week 6 | 卷积、池化、经典架构、残差 |
| 第 8-9 讲：RNN I/II | Week 7 | BPTT、双向/深层 RNN、LSTM、GRU |
| 第 10 讲：无监督学习 | Week 8 | Hopfield、Boltzmann 系、AE 家族 |
| 第 11 讲：前沿与局限 | Week 9 | Attention、VAE/GAN、深度 RL、GNN、局限 |
| 第 12 讲：行业应用 | Week 10 Day 1 | 框架趋势、MindSpore 主题；文字资料替代视频 |
| 实验 1-2：PyTorch 与前馈网络 | Week 4-5 | 两种训练写法、模型选择、正则化 |
| 实验 3：CNN | Week 6 | 手写卷积、分类、空洞卷积、ResNet |
| 实验 4：RNN | Week 7 | 序列处理、RNN/LSTM/GRU 对照 |
| 实验 5：综合选题（五选一） | Week 10-12 | 五题都了解，完成其中一题及报告 |

## 完成标准与后续

完成计划不等于精通所有模型。验收重点是能解释核心结构与训练目标、运行并分析小实验、分清训练集/验证集、写出综合实验的可复现报告。Hopfield、DBM/DBN、深度强化学习和图神经网络先做到课程级概念与小例子；它们各自深入都需要额外时间。进度不以日历硬推：如果基础或实验明显吃力，可延长到 14-16 周。

课程结束时，以能讲清主要知识点、独立运行并分析实验、提交可复现的综合实验报告为验收。进组后的具体研究方向以老师安排为准，暂不预设额外项目。

## 课程资料

- [课程主页：12 讲和 5 组实验](https://rmcong.github.io/proj_deep_learning_ProfessionalCourse.html)
- [课程安排 PDF](https://rmcong.github.io/DL%20Course%202022/schedule_p.pdf)
- [第 5 讲：优化与正则化 PDF](https://rmcong.github.io/DL%20Course%202022/chapter5.pdf)
- [第 10 讲：无监督学习 PDF](https://rmcong.github.io/DL%20Course%202022/chapter10.pdf)
- [第 11 讲：深度学习前沿 PDF](https://rmcong.github.io/DL%20Course%202022/chapter11.pdf)
- [实验 5：综合选题 PDF](https://rmcong.github.io/DL%20Course%202022/experiment5.pdf)
