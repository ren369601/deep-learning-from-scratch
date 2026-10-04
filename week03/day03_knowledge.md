# Week 3 / Day 3：Loss Function 与 Cross Entropy

模型完成前向传播后会产生预测，但训练还需要回答：

```text
预测结果距离正确答案有多远？
```

损失函数会把预测质量转换成一个数字：

```text
loss 小：预测较好
loss 大：预测较差
```

## 1. 不同任务使用不同损失

常见选择：

```text
回归任务：MSE
分类任务：Cross Entropy
```

回归预测连续数值，例如房价、温度、分数。

分类预测类别，例如图片属于猫、狗还是汽车，文本属于正面还是负面。

## 2. 回顾 MSE

```text
MSE = mean((predictions - targets)^2)
```

NumPy：

```python
loss = np.mean((predictions - targets) ** 2)
```

MSE 直接衡量预测数值和真实数值之间的距离，因此适合回归。

## 3. 分类模型的输出

多分类模型通常先输出 logits：

```text
logits = [2.0, 1.0, 0.0]
```

经过 Softmax 后得到概率：

```text
probabilities = [0.665, 0.245, 0.090]
```

如果真实类别是类别 0，那么我们最关心：

```text
模型给正确类别 0 分配了多少概率？
```

## 4. Cross Entropy 的核心

当真实类别使用整数下标表示时，单个样本的交叉熵可以写成：

```text
loss = -log(正确类别的预测概率)
```

例如真实类别是 0：

```text
probabilities = [0.8, 0.1, 0.1]
正确类别概率 = 0.8
loss = -log(0.8) ≈ 0.223
```

如果模型只给正确类别 0.1 的概率：

```text
loss = -log(0.1) ≈ 2.303
```

正确类别概率越高，loss 越小。

## 5. 为什么要使用负对数？

`-log(p)` 有很适合分类训练的性质：

```text
p 接近 1，loss 接近 0
p 变小，loss 增大
p 接近 0，loss 会非常大
```

大致关系：

| 正确类别概率 | Cross Entropy |
|---:|---:|
| 0.99 | 0.010 |
| 0.80 | 0.223 |
| 0.50 | 0.693 |
| 0.10 | 2.303 |
| 0.01 | 4.605 |

所以模型如果非常自信地预测错误，会受到明显惩罚。

## 6. 一批样本的 Cross Entropy

假设有 3 个样本、3 个类别：

```python
probabilities = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7],
])

targets = np.array([0, 1, 2])
```

正确类别概率分别是：

```text
0.8, 0.7, 0.7
```

整批 loss：

```text
mean([-log(0.8), -log(0.7), -log(0.7)])
```

## 7. 如何取出正确类别概率？

```python
batch_indices = np.arange(len(targets))
correct_probabilities = probabilities[batch_indices, targets]
```

其中：

```text
batch_indices = [0, 1, 2]
targets       = [0, 1, 2]
```

NumPy 会依次取：

```text
probabilities[0, 0]
probabilities[1, 1]
probabilities[2, 2]
```

## 8. 为什么需要 clip？

如果正确类别概率恰好是 0：

```text
log(0) -> 负无穷
```

因此计算前进行裁剪：

```python
safe_probabilities = np.clip(probabilities, 1e-12, 1.0)
```

这样最小概率不会低于 `1e-12`，可以避免无穷值。

## 9. 整数标签和 One-Hot 标签

整数标签：

```text
targets = [0, 2, 1]
```

One-hot 标签：

```text
[[1, 0, 0],
 [0, 0, 1],
 [0, 1, 0]]
```

可以使用：

```python
one_hot = np.eye(num_classes)[targets]
```

交叉熵也可以写成：

```python
loss = -np.mean(np.sum(one_hot * np.log(probabilities), axis=1))
```

本课程当前主要使用更简洁的整数标签。

## 10. Loss 和 Accuracy 不一样

Accuracy 只检查预测类别是否正确：

```python
predicted_classes = np.argmax(probabilities, axis=1)
accuracy = np.mean(predicted_classes == targets)
```

两个模型可能 accuracy 相同，但 Cross Entropy 不同：

```text
模型 A 正确类别概率为 0.51
模型 B 正确类别概率为 0.95
```

两者都预测正确，但模型 B 的 loss 更小，因为它对正确答案更有信心。

训练通常优化可求导的 loss，而 accuracy 主要用于观察模型表现。

## 11. 从 Logits 到 Loss

完整分类前向流程：

```text
X
-> Linear
-> ReLU
-> Linear
-> logits
-> Softmax
-> probabilities
-> Cross Entropy
```

今天会把 Day 2 的小型网络接上 targets，并计算分类 loss 与 accuracy。

## 12. 今日关键词

```text
loss function：损失函数
regression：回归
classification：分类
cross entropy：交叉熵
correct-class probability：正确类别概率
integer label：整数标签
one-hot label：独热标签
accuracy：准确率
```
