# 阶段梳理：从线性模型到两层 MLP（Week 1 至 Week 3 / Day 5）

先记住一条主线：

```text
数据 X + 参数 W、b
    -> 前向传播，得到预测
    -> 与真实答案 targets 比较，得到 loss
    -> 反向传播，得到参数的梯度
    -> 更新参数
    -> 重复，直到预测改善
```

目前我们已经完整训练过线性回归；两层 MLP 则完成了前向传播和标量反向传播练习。矩阵版 MLP 反向传播还没学到。

## 一、每一周解决了什么问题？

| 阶段 | 核心问题 | 学到的工具 |
|---|---|---|
| Week 1 | 一个模型如何根据输入给出预测？怎样判断好坏？ | Python、`y = wx + b`、批量样本、MSE |
| Week 2 | 如何高效处理一批数据，并自动调整参数？ | NumPy、shape、`@`、广播、梯度下降 |
| Week 3 Day 1-2 | 如何从线性模型组成神经网络？ | 神经元、Linear、ReLU、Sigmoid、Tanh、Softmax |
| Week 3 Day 3 | 分类预测的好坏如何衡量？ | logits、Cross Entropy、accuracy |
| Week 3 Day 4 | loss 如何影响前面的参数？ | 导数、链式法则、标量反向传播 |
| Week 3 Day 5 | 如何组装两层网络？ | `Linear -> ReLU -> Linear -> Softmax` |

## 二、先分清数据、参数和中间值

以一批数据经过两层 MLP 为例：

```text
X：输入数据，由数据集提供
targets：真实答案，由数据集提供
W1、b1、W2、b2：模型参数，训练时会更新
Z1、A1、logits、probabilities：前向传播产生的中间值或预测
loss：预测与真实答案的差距
grad_W1 等：loss 对参数的梯度，用来更新参数
```

`X` 和 `targets` 是要学习的例子；`W` 和 `b` 是模型要学会的东西。`Z1`、`A1` 等中间值每次前向传播都会重新计算。

## 三、一个神经元、一个线性层、一个 MLP

一个神经元接收一组特征，输出一个数：

```text
z = x @ w + b
```

一组神经元组成线性层，输出一组数：

```text
Z = X @ W + b
```

多层感知机（MLP）把线性层与激活函数连接起来：

```text
Z1 = X @ W1 + b1
A1 = ReLU(Z1)
logits = A1 @ W2 + b2
probabilities = Softmax(logits)
```

线性层的输出叫 `Z1`；经过 ReLU 的结果叫 `A1`。第二层输出的原始分类分数叫 `logits`，经过 Softmax 后才是概率。

## 四、Shape 是计算的说明书

设 `N` 是样本数、`D` 是输入特征数、`H` 是隐藏神经元数、`C` 是类别数：

```text
X                 (N, D)
W1                (D, H)
b1                (H,)
Z1 / A1           (N, H)
W2                (H, C)
b2                (C,)
logits            (N, C)
probabilities     (N, C)
targets           (N,)      # 整数类别标签
```

例如 `N=5, D=3, H=5, C=3`，最终 `probabilities.shape` 是 `(5, 3)`：5 个样本，各有 3 个类别概率。

矩阵乘法先看内侧维度：

```text
(N, D) @ (D, H) -> (N, H)
```

偏置 `(H,)` 广播到每个样本，因此结果仍是 `(N, H)`。

## 五、容易混淆的四组概念

### 1. `*`、`@`、`reshape`、`transpose`

```text
*：逐元素相乘；可以广播。
@：向量点积或矩阵乘法；检查内侧维度。
reshape：改变元素的组织形状；元素总数不变。
transpose：交换轴的位置；例如 (2, 3) -> (3, 2)。
```

`reshape` 和 `transpose` 的结果可能有相同 shape，但元素排列通常不同。

### 2. Logits、Softmax 概率、预测类别

```text
logits = [2, 1, 0]                    # 原始分数
probabilities = softmax(logits)      # 约 [0.665, 0.245, 0.090]
predicted_class = argmax(probabilities)  # 0
```

`argmax` 只选出概率最大的类别；它本身不能表达模型有多自信。

### 3. MSE、Cross Entropy、Accuracy

```text
MSE：衡量连续数值预测的误差，常用于回归。
Cross Entropy：衡量正确类别得到的概率，常用于分类训练。
Accuracy：统计有多少样本分类正确，常用于观察表现。
```

分类中，两个模型都可能预测正确、accuracy 相同；正确类别概率更高的模型通常 Cross Entropy 更小。

### 4. Forward、Backward、Update

```text
Forward：用当前参数算预测与 loss。
Backward：算 loss 对每个参数的梯度。
Update：根据梯度改变参数，例如 W -= lr * grad_W。
```

梯度不是新参数。它只告诉我们：参数稍微变化时，loss 会怎样变化。

## 六、反向传播怎样接到前向传播上？

标量例子：

```text
z = w*x + b
loss = (z - target)^2
```

前向：先算 `z`，再算 `loss`。

反向：先算 loss 对 `z` 的梯度，再沿路径算到参数：

```text
dLoss/dz = 2*(z - target)
dLoss/dw = dLoss/dz * x
dLoss/db = dLoss/dz
```

这就是链式法则：一个参数通过中间结果影响 loss，就把沿途的局部梯度相乘。

ReLU 的局部梯度：输入大于 0 时为 1，小于或等于 0 时按当前练习约定为 0。反向传播需要知道前向时 ReLU 看到了什么，因此 Day 5 在层里缓存了输入。

## 七、现在处于哪一步？

已经完成：

```text
线性回归：Forward -> MSE -> 梯度 -> 更新参数
两层 MLP：Forward -> Softmax 概率
标量两层网络：手算 Forward -> Backward -> 一次更新
```

下一步：把标量反向传播推广到一批样本和矩阵，训练两层 MLP。这里会复用 Day 5 保存的输入与隐藏层结果。

## 八、五分钟自测

先不看上文，试着回答：

1. `(4, 2) @ (2, 3)` 的结果 shape 是什么？
2. 为什么两层 Linear 中间要放 ReLU？
3. logits 与 probabilities 有什么区别？
4. 一个样本的正确类别概率从 `0.2` 增到 `0.9`，Cross Entropy 如何变化？
5. `loss.backward()` 将来要替我们做哪件事？
6. 为什么 Day 5 的 Linear 和 ReLU 要保存前向输入？

参考答案：`(4, 3)`；引入非线性；原始分数与概率；loss 变小；计算梯度；Backward 需要这些中间值。
