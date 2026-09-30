# 深度学习 24 周每日 / 每周详细计划

这个计划用于配合长期项目：

```text
从零开始学习深度学习，最终从零实现 Transformer，并具备论文阅读与复现能力。
```

建议节奏：

- 每周学习 5 天，留 1 天复习，留 1 天休息或补进度。
- 每天按照：知识讲解 → 示例代码 → 小练习 → 当天作业 → GitHub 积累。
- 每周结束后整理一次 `notes.md`，并提交到 GitHub。

## 总览

| 周数 | 阶段 | 本周主题 | 本周项目产出 |
|---|---|---|---|
| Week 1 | Python 基础 | Python、函数、class、线性模型 | 单输入 / 多输入线性模型 |
| Week 2 | NumPy 基础 | array、shape、矩阵运算 | NumPy Linear Regression |
| Week 3 | 数学基础 | 向量、矩阵、矩阵乘法 | 手写矩阵计算练习 |
| Week 4 | 微积分与梯度 | 导数、偏导、链式法则 | 手算 Gradient |
| Week 5 | 神经网络基础 | Neuron、Activation、Loss | NumPy 单神经元 |
| Week 6 | 反向传播 | Backpropagation、梯度下降 | NumPy MLP |
| Week 7 | PyTorch 基础 | Tensor、Autograd | PyTorch 线性回归 |
| Week 8 | PyTorch 训练流程 | Dataset、DataLoader、nn.Module | PyTorch MLP |
| Week 9 | CNN 入门 | 卷积、池化、feature map | MNIST CNN |
| Week 10 | CNN 深入 | 训练技巧、过拟合、调参 | 改进 MNIST / 小型 CNN |
| Week 11 | 图像分类 | CIFAR-10、数据增强 | CIFAR-10 CNN |
| Week 12 | CNN 项目整理 | 模型保存、日志、实验记录 | CNN 项目整理 |
| Week 13 | RNN | 序列数据、Hidden State | 简单序列预测 |
| Week 14 | LSTM | 长期依赖、文本分类 | LSTM 文本分类 |
| Week 15 | Attention 入门 | Q、K、V、Softmax | 手写 Scaled Dot-Product Attention |
| Week 16 | Multi-Head Attention | 多头注意力、mask | 手写 Multi-Head Attention |
| Week 17 | Transformer Encoder | Embedding、Position、Encoder Layer | Transformer Encoder |
| Week 18 | Transformer Decoder | Decoder、Mask、Cross Attention | Transformer Decoder |
| Week 19 | 完整 Transformer | Encoder-Decoder 组合 | 完整 Transformer 模型 |
| Week 20 | 训练 Transformer | 数据、训练循环、checkpoint | Transformer 训练脚本 |
| Week 21 | 论文阅读 | Attention Is All You Need | 论文阅读笔记 |
| Week 22 | 论文复现 | 对照论文复现模型结构 | Transformer 复现版本 |
| Week 23 | 实验与调参 | Ablation、超参数、可视化 | 实验记录 |
| Week 24 | GitHub 项目整理 | README、结构、总结 | 完整 GitHub 项目 |

## Week 1：Python 基础与线性模型

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 线性模型 `y = wx + b`，变量、函数、list、for、class | 单输入线性模型 | `week01/day01_python_linear_model.py` |
| Day 2 | 多输入线性模型，feature 和 weight 的对应关系 | `y = w1*x1 + w2*x2 + b` | `week01/day02_multi_feature_linear.py` |
| Day 3 | Python list 深入，嵌套 list，批量样本 | 多个样本批量预测 | `week01/day03_batch_prediction.py` |
| Day 4 | 函数拆分与代码结构 | 把预测、打印、测试拆成函数 | `week01/day04_functions_refactor.py` |
| Day 5 | class 加强，模型对象，参数保存 | 完整 `LinearModel` 类 | `week01/day05_linear_model_class.py` |
| Day 6 | 本周复习 | 重写 Day 1-5 的核心代码 | `week01/review.md` |
| Day 7 | 休息 / 补进度 | 整理笔记并提交 GitHub | GitHub commit |

## Week 2：NumPy 基础与 Linear Regression

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | NumPy array、shape、dtype | 创建向量和矩阵 | `week02/day01_numpy_array.py` |
| Day 2 | reshape、transpose、广播 | shape 变化练习 | `week02/day02_shape_ops.py` |
| Day 3 | 向量点积和矩阵乘法 | 手写 dot product，对比 NumPy | `week02/day03_dot_matmul.py` |
| Day 4 | 用 NumPy 写线性模型 | `y = XW + b` | `week02/day04_numpy_linear.py` |
| Day 5 | MSE Loss | 计算预测误差 | `week02/day05_mse_loss.py` |
| Day 6 | 梯度下降入门 | 手动更新 `w` 和 `b` | `week02/day06_gradient_descent.py` |
| Day 7 | 复习与整理 | 完成 NumPy Linear Regression | `week02/linear_regression_numpy.py` |

## Week 3：线性代数基础

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 标量、向量、矩阵 | 用 Python / NumPy 表示三者 | `week03/day01_vector_matrix.py` |
| Day 2 | 矩阵 shape 判断 | 判断矩阵能否相乘 | `week03/day02_shape_check.py` |
| Day 3 | 矩阵乘法 | 手算和代码计算矩阵乘法 | `week03/day03_matrix_multiply.py` |
| Day 4 | 转置、单位矩阵、逆矩阵 | NumPy 操作练习 | `week03/day04_matrix_ops.py` |
| Day 5 | dot product 与相似度 | 计算两个向量相似度 | `week03/day05_dot_similarity.py` |
| Day 6 | 线性变换直觉 | 用矩阵改变向量 | `week03/day06_linear_transform.py` |
| Day 7 | 复习 | 整理 shape 规则表 | `week03/review.md` |

## Week 4：导数、梯度与链式法则

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 导数直觉 | 画出简单函数变化 | `week04/day01_derivative_intro.py` |
| Day 2 | 偏导数 | 多变量函数求偏导 | `week04/day02_partial_derivative.md` |
| Day 3 | MSE 对 `w`、`b` 的梯度 | 手算线性回归梯度 | `week04/day03_linear_gradient.md` |
| Day 4 | 链式法则 | 手算复合函数梯度 | `week04/day04_chain_rule.md` |
| Day 5 | 梯度下降 | 代码模拟参数更新 | `week04/day05_gradient_descent.py` |
| Day 6 | 从 loss 到参数更新 | 连接预测、loss、gradient | `week04/day06_training_step.py` |
| Day 7 | 复习 | 梯度公式整理 | `week04/review.md` |

## Week 5：神经网络基础

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Neuron | 写一个单神经元 | `week05/day01_neuron.py` |
| Day 2 | Linear Layer | 多输入多输出线性层 | `week05/day02_linear_layer.py` |
| Day 3 | Activation Function | Sigmoid、Tanh、ReLU | `week05/day03_activation.py` |
| Day 4 | Loss Function | MSE、BCE、Cross Entropy 直觉 | `week05/day04_loss.py` |
| Day 5 | Forward Propagation | 完成一次前向传播 | `week05/day05_forward.py` |
| Day 6 | 简单分类任务 | 用 NumPy 做二分类 | `week05/day06_binary_classifier.py` |
| Day 7 | 复习 | 整理神经网络基本流程 | `week05/review.md` |

## Week 6：反向传播与 NumPy MLP

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Backpropagation 直觉 | 手算一个神经元梯度 | `week06/day01_backprop_intro.md` |
| Day 2 | Linear 层反向传播 | 手写 `dW`、`db`、`dX` | `week06/day02_linear_backward.py` |
| Day 3 | ReLU 反向传播 | 手写 ReLU backward | `week06/day03_relu_backward.py` |
| Day 4 | 两层 MLP | NumPy 实现 MLP forward | `week06/day04_mlp_forward.py` |
| Day 5 | MLP backward | 完成反向传播 | `week06/day05_mlp_backward.py` |
| Day 6 | 训练 MLP | 训练一个小分类器 | `week06/day06_train_mlp.py` |
| Day 7 | 复习 | 总结 forward / backward | `week06/review.md` |

## Week 7：PyTorch 基础

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Tensor | 创建和操作 Tensor | `week07/day01_tensor.py` |
| Day 2 | Tensor shape | reshape、view、transpose | `week07/day02_tensor_shape.py` |
| Day 3 | Autograd | `requires_grad` 和 `backward()` | `week07/day03_autograd.py` |
| Day 4 | PyTorch 线性回归 | 自动求梯度训练 `w`、`b` | `week07/day04_torch_linear_regression.py` |
| Day 5 | nn.Module | 自定义模型类 | `week07/day05_nn_module.py` |
| Day 6 | Optimizer | SGD、Adam 基本使用 | `week07/day06_optimizer.py` |
| Day 7 | 复习 | 对比 NumPy 和 PyTorch | `week07/review.md` |

## Week 8：PyTorch 训练流程

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Dataset | 自定义 Dataset | `week08/day01_dataset.py` |
| Day 2 | DataLoader | batch、shuffle | `week08/day02_dataloader.py` |
| Day 3 | Training Loop | 标准训练循环 | `week08/day03_training_loop.py` |
| Day 4 | Validation | 训练集 / 验证集 | `week08/day04_validation.py` |
| Day 5 | 保存模型 | `state_dict` 和 checkpoint | `week08/day05_checkpoint.py` |
| Day 6 | PyTorch MLP | 完整训练一个 MLP | `week08/day06_pytorch_mlp.py` |
| Day 7 | 复习 | 整理训练流程模板 | `week08/review.md` |

## Week 9：CNN 入门与 MNIST

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | CNN 直觉 | 卷积核如何看局部特征 | `week09/day01_cnn_intro.md` |
| Day 2 | Conv2d | 输入输出 shape 计算 | `week09/day02_conv2d.py` |
| Day 3 | Padding / Stride | 输出尺寸变化 | `week09/day03_padding_stride.py` |
| Day 4 | Pooling | MaxPool / AvgPool | `week09/day04_pooling.py` |
| Day 5 | MNIST CNN | 搭建 CNN 模型 | `week09/day05_mnist_cnn.py` |
| Day 6 | 训练 MNIST | 完成训练和验证 | `week09/day06_train_mnist.py` |
| Day 7 | 复习 | 记录 CNN shape 变化 | `week09/review.md` |

## Week 10：CNN 深入

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 过拟合与欠拟合 | 观察训练 / 验证 loss | `week10/day01_overfit_underfit.md` |
| Day 2 | Dropout | 加入 Dropout | `week10/day02_dropout.py` |
| Day 3 | BatchNorm | 使用 BatchNorm | `week10/day03_batchnorm.py` |
| Day 4 | 学习率 | 对比不同 learning rate | `week10/day04_learning_rate.py` |
| Day 5 | Optimizer 对比 | SGD vs Adam | `week10/day05_optimizer_compare.py` |
| Day 6 | 改进 CNN | 提升 MNIST 模型 | `week10/day06_improved_cnn.py` |
| Day 7 | 复习 | 整理调参经验 | `week10/review.md` |

## Week 11：CIFAR-10 图像分类

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | CIFAR-10 数据集 | 加载和可视化数据 | `week11/day01_cifar10_data.py` |
| Day 2 | 彩色图片输入 | 理解 `[B, C, H, W]` | `week11/day02_image_shape.py` |
| Day 3 | CIFAR CNN | 搭建更深 CNN | `week11/day03_cifar_cnn.py` |
| Day 4 | 数据增强 | RandomCrop、Flip | `week11/day04_augmentation.py` |
| Day 5 | 训练 CIFAR | 训练和验证 | `week11/day05_train_cifar.py` |
| Day 6 | 错误分析 | 查看分类错误样本 | `week11/day06_error_analysis.py` |
| Day 7 | 复习 | 整理 CIFAR 项目 | `week11/review.md` |

## Week 12：CNN 项目整理

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 项目结构 | 拆分 model、train、utils | `week12/day01_project_structure.md` |
| Day 2 | 配置管理 | 保存超参数 | `week12/day02_config.py` |
| Day 3 | 日志记录 | 记录 loss 和 accuracy | `week12/day03_logging.py` |
| Day 4 | 模型保存与加载 | checkpoint 管理 | `week12/day04_checkpoint.py` |
| Day 5 | README 初稿 | 写 CNN 项目说明 | `week12/README.md` |
| Day 6 | 整合代码 | 整理 CNN 项目 | `week12/cnn_project/` |
| Day 7 | GitHub 提交 | 提交 CNN 阶段成果 | GitHub commit |

## Week 13：RNN

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 序列数据 | 理解时间步 `t` | `week13/day01_sequence_data.md` |
| Day 2 | RNN 结构 | Hidden State | `week13/day02_rnn_intro.py` |
| Day 3 | 手写简单 RNN | NumPy / PyTorch 实现 | `week13/day03_simple_rnn.py` |
| Day 4 | 序列预测 | 预测简单数字序列 | `week13/day04_sequence_prediction.py` |
| Day 5 | 梯度消失 | RNN 的问题 | `week13/day05_vanishing_gradient.md` |
| Day 6 | PyTorch RNN | 使用 `nn.RNN` | `week13/day06_torch_rnn.py` |
| Day 7 | 复习 | 总结 RNN 输入输出 shape | `week13/review.md` |

## Week 14：LSTM

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | LSTM 直觉 | 门控机制 | `week14/day01_lstm_intro.md` |
| Day 2 | LSTM 输入输出 | 理解 hidden / cell state | `week14/day02_lstm_shape.py` |
| Day 3 | 文本表示 | token、vocab、embedding | `week14/day03_text_embedding.py` |
| Day 4 | LSTM 文本分类 | 搭建模型 | `week14/day04_lstm_classifier.py` |
| Day 5 | 训练文本分类 | 完成训练循环 | `week14/day05_train_text_classifier.py` |
| Day 6 | RNN vs LSTM vs Transformer | 对比三者 | `week14/day06_compare.md` |
| Day 7 | 复习 | 整理序列模型笔记 | `week14/review.md` |

## Week 15：Attention 入门

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Attention 直觉 | 查询和匹配 | `week15/day01_attention_intro.md` |
| Day 2 | Q / K / V | 理解三个向量角色 | `week15/day02_qkv.md` |
| Day 3 | Dot Product Attention | 计算相似度分数 | `week15/day03_dot_attention.py` |
| Day 4 | Softmax | 把分数变成权重 | `week15/day04_softmax.py` |
| Day 5 | Scaled Dot-Product Attention | 手写 Attention 公式 | `week15/day05_scaled_dot_product_attention.py` |
| Day 6 | Attention shape | 分析 `[B, T, D]` | `week15/day06_attention_shape.md` |
| Day 7 | 复习 | 整理 Attention 公式 | `week15/review.md` |

## Week 16：Multi-Head Attention

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 为什么需要多头 | 多角度关注信息 | `week16/day01_multi_head_intro.md` |
| Day 2 | 拆分 head | `[B, T, D]` 到 `[B, H, T, Dh]` | `week16/day02_split_heads.py` |
| Day 3 | 合并 head | `[B, H, T, Dh]` 到 `[B, T, D]` | `week16/day03_concat_heads.py` |
| Day 4 | Mask | padding mask / causal mask | `week16/day04_mask.py` |
| Day 5 | MultiHeadAttention 类 | 手写完整模块 | `week16/day05_multi_head_attention.py` |
| Day 6 | 测试 MHA | 检查 shape 和输出 | `week16/day06_test_mha.py` |
| Day 7 | 复习 | 整理 MHA shape 流程 | `week16/review.md` |

## Week 17：Transformer Encoder

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Embedding | token 到向量 | `week17/day01_embedding.py` |
| Day 2 | Positional Encoding | 位置信息 | `week17/day02_positional_encoding.py` |
| Day 3 | Feed Forward Network | FFN 模块 | `week17/day03_ffn.py` |
| Day 4 | LayerNorm | 归一化 | `week17/day04_layernorm.py` |
| Day 5 | Residual Connection | 残差连接 | `week17/day05_residual.py` |
| Day 6 | EncoderLayer | 组合 Encoder Layer | `week17/day06_encoder_layer.py` |
| Day 7 | Encoder | 堆叠多层 Encoder | `week17/day07_encoder.py` |

## Week 18：Transformer Decoder

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Decoder 结构 | 与 Encoder 的区别 | `week18/day01_decoder_intro.md` |
| Day 2 | Causal Mask | 防止看到未来 token | `week18/day02_causal_mask.py` |
| Day 3 | Masked Self-Attention | Decoder 第一层注意力 | `week18/day03_masked_self_attention.py` |
| Day 4 | Cross Attention | Decoder 关注 Encoder 输出 | `week18/day04_cross_attention.py` |
| Day 5 | DecoderLayer | 组合 Decoder Layer | `week18/day05_decoder_layer.py` |
| Day 6 | Decoder | 堆叠多层 Decoder | `week18/day06_decoder.py` |
| Day 7 | 复习 | 整理 Decoder shape 流程 | `week18/review.md` |

## Week 19：完整 Transformer

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Transformer 总结构 | Encoder + Decoder | `week19/day01_transformer_arch.md` |
| Day 2 | 输入输出接口 | source / target | `week19/day02_io_interface.py` |
| Day 3 | 完整模型类 | `Transformer` 类 | `week19/day03_transformer.py` |
| Day 4 | Mask 整合 | padding mask + causal mask | `week19/day04_masks.py` |
| Day 5 | 前向传播测试 | 检查完整 shape | `week19/day05_forward_test.py` |
| Day 6 | 参数初始化 | 初始化和稳定训练 | `week19/day06_init.py` |
| Day 7 | 复习 | 整理完整结构图 | `week19/review.md` |

## Week 20：训练 Transformer

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 任务选择 | 小型翻译 / toy dataset | `week20/day01_task_setup.md` |
| Day 2 | 数据预处理 | tokenization、vocab | `week20/day02_preprocess.py` |
| Day 3 | Dataset / DataLoader | 构建训练数据 | `week20/day03_dataset.py` |
| Day 4 | Loss 和 Optimizer | Cross Entropy、Adam | `week20/day04_loss_optimizer.py` |
| Day 5 | 训练循环 | 完成训练脚本 | `week20/day05_train.py` |
| Day 6 | 推理生成 | greedy decoding | `week20/day06_inference.py` |
| Day 7 | 复习 | 保存模型和训练记录 | `week20/review.md` |

## Week 21：Attention Is All You Need 论文阅读

| 天数 | 学习内容 | 代码 / 笔记 | 产出文件 |
|---|---|---|---|
| Day 1 | 论文背景和摘要 | 读 Abstract / Introduction | `week21/day01_paper_intro.md` |
| Day 2 | 模型结构 | 读 Model Architecture | `week21/day02_architecture.md` |
| Day 3 | Attention 公式 | 对照代码理解公式 | `week21/day03_attention_formula.md` |
| Day 4 | Multi-Head Attention | 对照 MHA 实现 | `week21/day04_multi_head.md` |
| Day 5 | Position Encoding / FFN | 理解辅助模块 | `week21/day05_components.md` |
| Day 6 | 训练细节 | optimizer、label smoothing | `week21/day06_training_details.md` |
| Day 7 | 论文总结 | 写完整阅读笔记 | `week21/paper_notes.md` |

## Week 22：论文复现

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 对照论文检查模块 | 检查模块完整性 | `week22/day01_module_check.md` |
| Day 2 | 复现模型参数 | d_model、heads、layers | `week22/day02_config.py` |
| Day 3 | 复现训练设置 | optimizer、scheduler | `week22/day03_training_setup.py` |
| Day 4 | 跑通小实验 | 小数据训练 | `week22/day04_small_experiment.py` |
| Day 5 | Debug 训练问题 | loss、shape、mask | `week22/day05_debug.md` |
| Day 6 | 整理复现代码 | 形成 `src/` 结构 | `week22/src/` |
| Day 7 | 复习 | 写复现总结 | `week22/reproduction_notes.md` |

## Week 23：实验、调参与 Ablation

| 天数 | 学习内容 | 代码 / 笔记 | 产出文件 |
|---|---|---|---|
| Day 1 | 实验记录方法 | 建立实验表格 | `week23/day01_experiment_log.md` |
| Day 2 | 学习率实验 | 对比不同 learning rate | `week23/day02_lr_experiment.py` |
| Day 3 | head 数量实验 | 对比不同 heads | `week23/day03_heads_experiment.py` |
| Day 4 | layer 数量实验 | 对比不同层数 | `week23/day04_layers_experiment.py` |
| Day 5 | dropout 实验 | 对比不同 dropout | `week23/day05_dropout_experiment.py` |
| Day 6 | 结果分析 | 写实验结论 | `week23/day06_analysis.md` |
| Day 7 | 复习 | 整理 ablation 表格 | `week23/ablation_summary.md` |

## Week 24：GitHub 完整项目整理

| 天数 | 学习内容 | 代码 / 文档 | 产出文件 |
|---|---|---|---|
| Day 1 | 项目结构整理 | 统一目录 | `src/`, `experiments/`, `docs/` |
| Day 2 | README | 写项目介绍、安装、运行方式 | `README.md` |
| Day 3 | 代码清理 | 删除临时文件、统一命名 | cleaned code |
| Day 4 | 注释和文档 | 给核心模块补说明 | `docs/` |
| Day 5 | 示例运行 | 提供 demo 脚本 | `examples/` |
| Day 6 | 最终检查 | 从零运行项目 | final test |
| Day 7 | 最终提交 | 推送 GitHub，写总结 | final GitHub commit |

## 每天学习模板

每天建议新建或更新这些内容：

```text
weekXX/
  dayXX_knowledge.md      # 知识讲解
  dayXX_python_syntax.md  # 如果当天涉及新语法，就单独记录
  dayXX_xxx.py            # 代码练习
  notes.md                # 当天或本周笔记
```

每天结束前检查：

```text
1. 今天的知识点能不能用自己的话讲出来？
2. 示例代码能不能独立重写？
3. 小练习有没有跑通？
4. 今天的 notes.md 有没有更新？
5. 是否需要提交 GitHub？
```

## GitHub 提交建议

每天或每周可以提交一次：

```powershell
git add .
git commit -m "Complete week X day Y"
git push
```

阶段性提交可以使用：

```powershell
git commit -m "Complete Python linear model practice"
git commit -m "Add NumPy linear regression"
git commit -m "Implement multi-head attention"
git commit -m "Implement transformer from scratch"
```
