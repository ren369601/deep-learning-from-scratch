# 深度学习 12 周压缩学习计划

这个版本把原来的 24 周计划压缩到约 3 个月。

目标保持不变：

```text
从零开始学习深度学习，最终能够从零实现 Transformer，并具备论文阅读与复现能力。
```

压缩原则：

- Python 和 NumPy 只学深度学习必需部分。
- CNN、RNN、LSTM 只学核心思想和必要代码，不做长时间深入。
- 把主要时间留给 PyTorch、反向传播、Attention、Transformer 和论文复现。
- 每周建议学习 5-6 天，每天 1.5-2.5 小时。
- 每周至少提交一次 GitHub。

## 12 周总览

| 周数 | 阶段 | 本周主题 | 本周项目产出 |
|---|---|---|---|
| Week 1 | Python 基础 | Python、函数、class、线性模型 | 单输入 / 多输入线性模型 |
| Week 2 | NumPy + 数学基础 | array、shape、矩阵乘法、梯度直觉 | NumPy Linear Regression |
| Week 3 | 神经网络基础 | Neuron、Activation、Loss、Backpropagation | NumPy 两层 MLP |
| Week 4 | PyTorch 基础 | Tensor、Autograd、nn.Module、Optimizer | PyTorch MLP |
| Week 5 | 训练流程 + CNN | Dataset、DataLoader、训练循环、CNN | MNIST CNN |
| Week 6 | 序列模型速览 | RNN、LSTM、Embedding、文本分类直觉 | 简单 LSTM 文本分类 |
| Week 7 | Attention | Q/K/V、Softmax、Scaled Dot-Product Attention | 手写 Attention |
| Week 8 | Multi-Head Attention | 多头、mask、shape 变化 | 手写 MultiHeadAttention |
| Week 9 | Transformer Encoder | Embedding、Position、FFN、LayerNorm、Residual | Transformer Encoder |
| Week 10 | Transformer Decoder + 完整模型 | Decoder、Causal Mask、Cross Attention | 完整 Transformer |
| Week 11 | 论文阅读与复现 | Attention Is All You Need | 论文笔记 + 对照复现 |
| Week 12 | 项目整理与实验 | 训练、调参、README、GitHub 整理 | 完整 GitHub 项目 |

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

## Week 5：训练流程与 CNN

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Dataset、DataLoader、batch | 自定义 Dataset | `week05/day01_dataset_dataloader.py` |
| Day 2 | CNN 直觉、Conv2d、Pooling | 输入输出 shape 计算 | `week05/day02_cnn_intro.py` |
| Day 3 | MNIST CNN | 搭建 CNN 模型 | `week05/day03_mnist_cnn.py` |
| Day 4 | 训练 MNIST | 完成训练和验证 | `week05/day04_train_mnist.py` |
| Day 5 | 过拟合、Dropout、学习率 | 改进 CNN | `week05/day05_improve_cnn.py` |
| Day 6 | CNN 阶段整理 | README 和结果记录 | `week05/review.md` |
| Day 7 | 休息 / 补进度 | GitHub 提交 | GitHub commit |

## Week 6：序列模型速览

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 序列数据、time step、hidden state | 简单序列预测 | `week06/day01_sequence_intro.py` |
| Day 2 | RNN 直觉 | 使用 `nn.RNN` | `week06/day02_rnn.py` |
| Day 3 | LSTM 直觉 | 使用 `nn.LSTM` | `week06/day03_lstm.py` |
| Day 4 | Embedding | token 到向量 | `week06/day04_embedding.py` |
| Day 5 | 简单文本分类 | LSTM 文本分类小例子 | `week06/day05_lstm_text_classifier.py` |
| Day 6 | 为什么需要 Transformer | RNN/LSTM vs Transformer | `week06/day06_why_transformer.md` |
| Day 7 | 复习 | 整理序列模型核心概念 | `week06/review.md` |

## Week 7：Attention

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Attention 直觉 | 查询、匹配、加权求和 | `week07/day01_attention_intro.md` |
| Day 2 | Q / K / V | 理解三个角色 | `week07/day02_qkv.md` |
| Day 3 | Dot Product Attention | 计算相似度分数 | `week07/day03_dot_attention.py` |
| Day 4 | Softmax | 把分数变成权重 | `week07/day04_softmax.py` |
| Day 5 | Scaled Dot-Product Attention | 手写 Attention 公式 | `week07/day05_scaled_dot_product_attention.py` |
| Day 6 | Attention shape | 分析 `[B, T, D]` | `week07/day06_attention_shape.md` |
| Day 7 | 复习 | 整理 Attention 公式 | `week07/review.md` |

## Week 8：Multi-Head Attention 与 Mask

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | 为什么需要多头 | 多个 head 的直觉 | `week08/day01_multi_head_intro.md` |
| Day 2 | split heads | `[B, T, D] → [B, H, T, Dh]` | `week08/day02_split_heads.py` |
| Day 3 | concat heads | `[B, H, T, Dh] → [B, T, D]` | `week08/day03_concat_heads.py` |
| Day 4 | Padding Mask | 屏蔽 padding | `week08/day04_padding_mask.py` |
| Day 5 | Causal Mask | 屏蔽未来 token | `week08/day05_causal_mask.py` |
| Day 6 | MultiHeadAttention 类 | 手写完整 MHA | `week08/day06_multi_head_attention.py` |
| Day 7 | 复习 | 测试 MHA shape | `week08/review.md` |

## Week 9：Transformer Encoder

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Embedding、Positional Encoding | token 和位置表示 | `week09/day01_embedding_position.py` |
| Day 2 | Feed Forward Network | FFN 模块 | `week09/day02_ffn.py` |
| Day 3 | LayerNorm、Residual | 残差和归一化 | `week09/day03_norm_residual.py` |
| Day 4 | EncoderLayer | 组合 MHA + FFN | `week09/day04_encoder_layer.py` |
| Day 5 | Encoder | 堆叠多层 Encoder | `week09/day05_encoder.py` |
| Day 6 | Encoder 测试 | shape 检查和单元测试 | `week09/day06_test_encoder.py` |
| Day 7 | 复习 | 整理 Encoder 结构 | `week09/review.md` |

## Week 10：Transformer Decoder 与完整模型

| 天数 | 学习内容 | 代码练习 | 产出文件 |
|---|---|---|---|
| Day 1 | Decoder 结构 | 与 Encoder 区别 | `week10/day01_decoder_intro.md` |
| Day 2 | Masked Self-Attention | Decoder 自注意力 | `week10/day02_masked_self_attention.py` |
| Day 3 | Cross Attention | Decoder 关注 Encoder 输出 | `week10/day03_cross_attention.py` |
| Day 4 | DecoderLayer / Decoder | 组合 Decoder | `week10/day04_decoder.py` |
| Day 5 | 完整 Transformer | Encoder + Decoder | `week10/day05_transformer.py` |
| Day 6 | 前向传播测试 | 检查完整 shape | `week10/day06_forward_test.py` |
| Day 7 | 复习 | 整理完整 Transformer 流程 | `week10/review.md` |

## Week 11：论文阅读与复现

| 天数 | 学习内容 | 代码 / 笔记 | 产出文件 |
|---|---|---|---|
| Day 1 | 读 Abstract / Introduction | 理解论文动机 | `week11/day01_paper_intro.md` |
| Day 2 | 读 Model Architecture | 对照自己实现的结构 | `week11/day02_architecture.md` |
| Day 3 | 读 Attention 公式 | 对照 Attention 代码 | `week11/day03_attention_formula.md` |
| Day 4 | 读 Multi-Head / FFN / Position | 对照模块代码 | `week11/day04_components.md` |
| Day 5 | 读训练细节 | optimizer、scheduler、label smoothing | `week11/day05_training_details.md` |
| Day 6 | 对照论文复现 | 调整命名和结构 | `week11/day06_reproduction.py` |
| Day 7 | 论文总结 | 写完整论文阅读笔记 | `week11/paper_notes.md` |

## Week 12：训练、实验与 GitHub 项目整理

| 天数 | 学习内容 | 代码 / 文档 | 产出文件 |
|---|---|---|---|
| Day 1 | 小型任务训练 Transformer | toy dataset / 小翻译任务 | `week12/day01_train_transformer.py` |
| Day 2 | 推理生成 | greedy decoding | `week12/day02_inference.py` |
| Day 3 | 实验记录 | learning rate、layers、heads | `week12/day03_experiment_log.md` |
| Day 4 | Debug 和调参 | loss、mask、shape 问题 | `week12/day04_debug_notes.md` |
| Day 5 | 整理项目结构 | `src/`、`docs/`、`examples/` | project structure |
| Day 6 | 写 README | 安装、运行、项目说明 | `README.md` |
| Day 7 | 最终检查与提交 | 从零运行项目并推送 | final GitHub commit |

## 每天学习模板

每天建议保持这个结构：

```text
weekXX/
  dayXX_knowledge.md
  dayXX_python_syntax.md  # 如果当天有新语法
  dayXX_xxx.py
  notes.md
```

每天结束前检查：

```text
1. 今天的知识点能不能用自己的话讲出来？
2. 示例代码能不能独立重写？
3. 小练习有没有跑通？
4. 今天的 notes.md 有没有更新？
5. 是否需要提交 GitHub？
```

## 压缩版注意事项

12 周版本节奏更快，所以要注意：

- 反向传播、shape、Attention、Mask 是关键点，不能为了赶进度跳过。
- CNN、RNN、LSTM 只作为理解深度学习和序列建模的桥梁，不做过多扩展。
- 如果某一周明显吃力，就把 12 周计划延长到 14-16 周，不影响最终目标。
- 最终目标不是“赶完表格”，而是能自己实现、解释和 debug Transformer。
