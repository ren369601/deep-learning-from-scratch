"""Week 3 / Day 3: MSE and Cross Entropy losses."""

import numpy as np


def softmax(logits):
    """Apply stable Softmax along the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

# 算MSE
def mean_squared_error(predictions, targets):
    """Return MSE for regression outputs with matching shapes."""
    # TODO:
    # 1. Validate matching shapes.
    assert predictions.shape == targets.shape, "Shapes must match"
    # 2. Return mean squared error as a Python float.
    mse = np.mean((predictions - targets) ** 2)
    return float(mse)

def cross_entropy(probabilities, targets):
    """Return mean Cross Entropy for integer class labels."""
    # TODO:
    # 1. Validate probabilities is 2D and targets is 1D.
    assert probabilities.ndim == 2, "probabilities must be 2D"
    assert targets.ndim == 1, "targets must be 1D"
    # 2. Validate batch sizes match.
    batch_size = probabilities.shape[0]
    assert len(targets) == batch_size, "Batch size mismatch"
    # 3. Clip probabilities to [1e-12, 1.0]. 防止log(0)无穷大
    probs_clipped = np.clip(probabilities, 1e-12, 1.0)
    # 4. Select each sample's correct-class probability.
    batch_indices = np.arange(batch_size)
    correct_p = probs_clipped[batch_indices, targets]
    # 5. Return the mean negative log probability as a Python float.
    loss = -np.mean(np.log(correct_p))
    return float(loss)

# 整合两部分
def cross_entropy_from_logits(logits, targets):
    """Apply Softmax and return both probabilities and Cross Entropy."""
    # TODO:
    # 1. Calculate probabilities with softmax().
    probabilities = softmax(logits)
    # 2. Calculate loss with cross_entropy().
    loss = cross_entropy(probabilities, targets)
    # 3. Return probabilities and loss.
    return probabilities, loss


def accuracy(probabilities, targets):
    """Return classification accuracy for integer labels."""
    # TODO:
    # 1. Get predicted classes with argmax(axis=1).
    # 2. Return the mean of predicted_classes == targets as a float.
    # 1. Get predicted classes with argmax(axis=1).
    predicted_classes = np.argmax(probabilities, axis=1)
    # 2. Return the mean of predicted_classes == targets as a float.
    acc = np.mean(predicted_classes == targets)
    return float(acc)


def demo_negative_log():
    """Show how correct-class probability changes the loss."""
    correct_probabilities = np.array([0.99, 0.8, 0.5, 0.1, 0.01])
    losses = -np.log(correct_probabilities)

    for probability, loss in zip(correct_probabilities, losses):
        print(f"probability={probability:.2f}, loss={loss:.4f}")


def practice_mse():
    """练习 1：为回归预测计算 MSE。"""
    predictions = np.array([2.5, 5.0, 8.5, 10.0])
    targets = np.array([3.0, 5.0, 7.0, 9.0])

    # TODO:
    # 1. 调用 mean_squared_error()。
    # 2. 打印 predictions、targets 和 loss。
    # 3. 说明为什么这个任务适合使用 MSE。
    # 1. 调用 mean_squared_error()。
    loss = mean_squared_error(predictions, targets)
    # 2. 打印 predictions、targets 和 loss。
    print("predictions:", predictions)
    print("targets:    ", targets)
    print("MSE loss:   ", loss)
    # 3. 说明为什么这个任务适合使用 MSE。
    print("MSE适合回归：回归输出是连续数值，MSE衡量预测值和真实值之间的距离")


def practice_cross_entropy():
    """练习 2：计算一批分类概率的 Cross Entropy。"""
    probabilities = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ], dtype=np.float32)
    targets = np.array([0, 1, 2])

    # TODO:
    # 1. 取出并打印每个样本的正确类别概率。
    # 2. 调用 cross_entropy() 并打印 loss。
    # 3. 调用 accuracy() 并打印准确率。
    # 1. 取出并打印每个样本的正确类别概率。
    batch_idx = np.arange(len(targets))
    correct_p = probabilities[batch_idx, targets]
    print("每个样本正确类别的概率：", correct_p)

    # 2. 调用 cross_entropy() 并打印 loss。
    loss = cross_entropy(probabilities, targets)
    print("Cross Entropy loss:", loss)

    # 3. 调用 accuracy() 并打印准确率。
    acc = accuracy(probabilities, targets)
    print("Accuracy:", acc)


def practice_compare_confidence():
    """练习 3：比较准确率相同但置信度不同的模型。"""
    targets = np.array([0, 1, 2])
    probabilities_a = np.array([
        [0.51, 0.25, 0.24],
        [0.24, 0.51, 0.25],
        [0.25, 0.24, 0.51],
    ])
    probabilities_b = np.array([
        [0.95, 0.03, 0.02],
        [0.02, 0.95, 0.03],
        [0.03, 0.02, 0.95],
    ])

    # TODO:
    # 1. 分别计算两个模型的 Cross Entropy 和 accuracy。
    # 2. 打印所有结果。
    # 3. 解释为什么 accuracy 相同而 loss 不同。
    ce_a = cross_entropy(probabilities_a, targets)
    acc_a = accuracy(probabilities_a, targets)
    ce_b = cross_entropy(probabilities_b, targets)
    acc_b = accuracy(probabilities_b, targets)
    print("Model A: CE loss =", ce_a, "Acc =", acc_a)
    print("Model B: CE loss =", ce_b, "Acc =", acc_b)
    print("解释：两个模型预测类别全部猜对，准确率都是1。")
    print("但是模型B对正确类别的置信度更高，所以交叉熵loss更小。")
    print("准确率只看预测类别是否正确；交叉熵还看模型自信程度。")


def homework_classification_loss():
    """作业：完成小型分类网络并评估 loss 与 accuracy。"""
    X = np.array([
        [1.0, 2.0],
        [2.0, -1.0],
        [-1.0, 1.0],
        [0.0, 3.0],
    ], dtype=np.float32)
    W1 = np.array([
        [0.5, -1.0, 2.0],
        [1.5, 0.5, -0.5],
    ], dtype=np.float32)
    b1 = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    W2 = np.array([
        [1.0, -1.0],
        [0.5, 2.0],
        [-1.5, 0.5],
    ], dtype=np.float32)
    b2 = np.array([0.5, -0.5], dtype=np.float32)
    targets = np.array([0, 1, 1, 0])

    # TODO:
    # 1. hidden = ReLU(X @ W1 + b1)，可直接使用 np.maximum()。
    # 2. logits = hidden @ W2 + b2。
    # 3. 调用 cross_entropy_from_logits() 得到 probabilities 和 loss。
    # 4. 计算 accuracy。
    # 5. 打印所有重要 shape、probabilities、targets、loss 和 accuracy。
    # 6. 打印预测类别，并检查它是否与 targets 一致。

    # 1. hidden = ReLU(X @ W1 + b1)，可直接使用 np.maximum()。
    hidden_raw = X @ W1 + b1
    hidden = np.maximum(hidden_raw, 0.0)  # ReLU
    # 2. logits = hidden @ W2 + b2。
    logits = hidden @ W2 + b2
    # 3. 调用 cross_entropy_from_logits() 得到 probabilities 和 loss。
    probabilities, loss = cross_entropy_from_logits(logits, targets)
    # 4. 计算 accuracy。
    acc = accuracy(probabilities, targets)
    # 5. 打印所有重要 shape、probabilities、targets、loss 和 accuracy。
    print("X shape:", X.shape)
    print("W1 shape:", W1.shape)
    print("hidden shape:", hidden.shape)
    print("W2 shape:", W2.shape)
    print("logits shape:", logits.shape)
    print("probabilities:\n", probabilities)
    print("targets:", targets)
    print("Cross Entropy loss =", loss)
    print("Accuracy =", acc)
    # 6. 打印预测类别，并检查它是否与 targets 一致。
    pred_classes = np.argmax(probabilities, axis=1)
    print("预测类别:", pred_classes)
    print("预测是否等于target：", pred_classes == targets)


if __name__ == "__main__":
    print("--- 负对数示例 ---")
    demo_negative_log()

    print("\n--- 练习 1：MSE ---")
    practice_mse()

    print("\n--- 练习 2：Cross Entropy ---")
    practice_cross_entropy()

    print("\n--- 练习 3：置信度比较 ---")
    practice_compare_confidence()

    print("\n--- 作业：分类网络损失 ---")
    homework_classification_loss()
