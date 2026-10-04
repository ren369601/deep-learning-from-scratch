# Week 3 / Day 2：激活函数

昨天实现了线性层：

```text
Z = X @ W + b
```

今天在线性层后加入激活函数：

```text
A = activation(Z)
```

这一步让神经网络能够学习非线性关系。

## 1. 为什么需要激活函数？

假设连续使用两个线性层：

```text
H = X @ W1 + b1
Y = H @ W2 + b2
```

代入后：

```text
Y = (X @ W1 + b1) @ W2 + b2
  = X @ (W1 @ W2) + (b1 @ W2 + b2)
```

它仍然可以合并成一个新的线性层：

```text
Y = X @ W_new + b_new
```

因此，无论叠多少个纯线性层，整体仍然只是线性模型。

加入非线性激活函数后：

```text
H = activation(X @ W1 + b1)
Y = H @ W2 + b2
```

这时两个线性层无法简单合并，网络才有能力表示更复杂的关系。

## 2. ReLU

ReLU 的公式：

```text
ReLU(x) = max(0, x)
```

```text
x < 0  -> 0
x >= 0 -> x
```

NumPy：

```python
def relu(x):
    return np.maximum(0, x)
```

例子：

```text
输入：[-2, -1, 0, 1, 2]
输出：[ 0,  0, 0, 1, 2]
```

ReLU 是隐藏层中最常见的激活函数之一，计算简单，并且正数区域的梯度不会快速变小。

## 3. Sigmoid

Sigmoid 公式：

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

输出范围：

```text
(0, 1)
```

特点：

```text
很大的负数接近 0
x = 0 时等于 0.5
很大的正数接近 1
```

Sigmoid 常用于二分类输出或门控结构。它在数值很大或很小时会进入饱和区，梯度可能变得很小。

## 4. Tanh

Tanh 输出范围：

```text
(-1, 1)
```

```python
def tanh(x):
    return np.tanh(x)
```

特点：

```text
x = 0 时输出 0
负输入得到负输出
正输入得到正输出
输出以 0 为中心
```

Tanh 曾广泛用于 RNN，但同样可能在较大绝对值区域出现梯度变小的问题。

## 5. Softmax

Softmax 把一组分数转换成概率分布：

```text
每个结果都在 0 到 1 之间
同一个样本的所有结果相加等于 1
较大的输入得到较大的概率
```

公式：

```text
softmax(zi) = exp(zi) / sum(exp(zj))
```

例如三个类别的分数：

```text
logits = [2.0, 1.0, 0.0]
```

Softmax 会把它们转换成类似：

```text
[0.665, 0.245, 0.090]
```

三个概率相加为 1。

## 6. 什么是 Logits？

分类模型最后一个线性层直接输出的原始分数通常叫 logits：

```text
logits = X @ W + b
```

logits 不是概率：

```text
可以是负数
可以大于 1
所有值不需要相加为 1
```

经过 Softmax 后才得到多分类概率：

```text
probabilities = softmax(logits)
```

## 7. Softmax 为什么要减去最大值？

直接计算 `exp(1000)` 会产生数值溢出。

稳定写法：

```python
shifted = logits - np.max(logits, axis=-1, keepdims=True)
exp_values = np.exp(shifted)
probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
```

给所有 logits 同时减去同一个数不会改变 Softmax 概率，但能避免指数值过大。

## 8. Softmax 的 Axis

批量分类时：

```text
logits.shape = (batch_size, num_classes)
```

每个样本应该单独得到一组概率，所以沿最后一个维度归一化：

```python
softmax(logits, axis=-1)
```

```text
(4, 3) -> 4 个样本，每个样本有 3 个类别概率
```

每一行的和应该接近 1。

## 9. 四种函数对比

| 函数 | 输出范围 | 常见用途 |
|---|---|---|
| ReLU | `[0, +∞)` | 隐藏层 |
| Sigmoid | `(0, 1)` | 二分类输出、门控 |
| Tanh | `(-1, 1)` | RNN、需要零中心输出的场景 |
| Softmax | 每项 `(0, 1)`，总和为 1 | 多分类输出 |

## 10. 一个两层网络的前向传播

```python
hidden_linear = X @ W1 + b1
hidden = relu(hidden_linear)
logits = hidden @ W2 + b2
probabilities = softmax(logits)
```

流程：

```text
输入
-> 第一层 Linear
-> ReLU
-> 第二层 Linear
-> Softmax
-> 类别概率
```

这已经是一个最小多层神经网络的前向传播。

## 11. 与 Transformer 的关系

Transformer 的 Feed Forward Network 通常包含：

```text
Linear -> 激活函数 -> Linear
```

原论文使用 ReLU，很多现代 Transformer 使用 GELU、SwiGLU 等激活函数。今天理解 ReLU 后，后续学习这些函数会更容易。

## 12. 今日关键词

```text
activation function：激活函数
nonlinearity：非线性
ReLU
Sigmoid
Tanh
Softmax
logits：分类模型的原始分数
probability distribution：概率分布
```
