"""Week 3 / Day 2: ReLU, Sigmoid, Tanh, and Softmax."""

import numpy as np


def relu(x):
    """Apply ReLU element-wise."""
    # TODO: Return max(0, x) for every element.
    return np.maximum(0,x)


def sigmoid(x):
    """Apply a numerically safe Sigmoid element-wise."""
    # TODO:
    # 1. Clip x to the range [-500, 500].
    # 2. Return 1 / (1 + exp(-x)).
    x_clipped = np.clip(x,-500,500)
    return 1.0/(1.0+np.exp(-x_clipped))


def tanh(x):
    """Apply Tanh element-wise."""
    # TODO: Use np.tanh(x).
    return np.tanh(x)


def softmax(logits):
    """Apply numerically stable Softmax along the last axis."""
    # TODO:
    # 1. Subtract the maximum along axis=-1 with keepdims=True.
    # 2. Calculate exponentials.
    # 3. Divide by their sum along axis=-1 with keepdims=True.
    max_vals = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_vals
    # 2. Calculate exponentials.
    exp_values = np.exp(shifted)
    # 3. Divide by their sum along axis=-1 with keepdims=True.
    sum_exp = np.sum(exp_values, axis=-1, keepdims=True)
    return exp_values / sum_exp


def demo_why_nonlinearity():
    """Show that two linear layers can be merged into one."""
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    W1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b1 = np.array([0.5, -0.5], dtype=np.float32)
    W2 = np.array([[2.0], [-1.0]], dtype=np.float32)
    b2 = np.array([1.0], dtype=np.float32)

    two_layers = (X @ W1 + b1) @ W2 + b2

    combined_weights = W1 @ W2
    combined_bias = b1 @ W2 + b2
    one_layer = X @ combined_weights + combined_bias

    print("two linear layers:", two_layers)
    print("combined one layer:", one_layer)
    print("same result:", np.allclose(two_layers, one_layer))


def practice_activation_values():
    """练习 1：比较 ReLU、Sigmoid 和 Tanh。"""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    # TODO:
    # 1. 分别调用 relu、sigmoid、tanh。
    r = relu(x)
    s = sigmoid(x)
    t = tanh(x)
    # 2. 打印输入和三个输出。
    print("输入 x:        ", x)
    print("ReLU:          ", r)
    print("Sigmoid:       ", s)
    print("Tanh:          ", t)
    # 3. 说明每个函数的输出范围。
    print("\n输出范围：")
    print("ReLU:  [0, +∞)")
    print("Sigmoid: (0, 1)")
    print("Tanh: (-1, 1)")



def practice_softmax_vector():
    """练习 2：把一组 logits 转换成概率。"""
    logits = np.array([2.0, 1.0, 0.0], dtype=np.float32)

    # TODO:
    # 1. 调用 softmax(logits)。
    probabilities = softmax(logits)
    # 2. 打印 probabilities。
    print("probabilities: ", probabilities)
    # 3. 打印 probabilities.sum()。
    print("sum:           ", probabilities.sum())
    # 4. 找出概率最大的类别下标，使用 np.argmax()。
    print("argmax index:  ", np.argmax(probabilities))



def practice_softmax_batch():
    """练习 3：对一批样本分别计算 Softmax。"""
    logits = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 3.0, 2.0],
        [1000.0, 1001.0, 999.0],
    ], dtype=np.float32)

    # TODO:
    # 1. 调用 softmax(logits)。
    probabilities = softmax(logits)
    # 2. 打印 logits.shape 和 probabilities.shape。
    print("logits.shape:       ", logits.shape)
    print("probabilities.shape:", probabilities.shape)
    # 3. 打印 probabilities。
    print("probabilities:\n", probabilities)
    # 4. 打印每一行的概率和。
    row_sum = np.sum(probabilities, axis=1)
    print("每一行的和：", row_sum)
    # 5. 使用 np.allclose() 验证每一行的和都接近 1。
    all_ok = np.allclose(row_sum, np.ones_like(row_sum))
    print("所有行和≈1？", all_ok)



def homework_tiny_network():
    """作业：完成 Linear -> ReLU -> Linear -> Softmax 前向传播。"""
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

    # TODO:
    # 1. hidden_linear = X @ W1 + b1。
    hidden_linear = X @ W1 + b1
    # 2. hidden = relu(hidden_linear)。
    hidden = relu(hidden_linear)
    # 3. logits = hidden @ W2 + b2。
    logits = hidden @ W2 + b2
    # 4. probabilities = softmax(logits)。
    probabilities = softmax(logits)
    # 5. 打印每一步的内容和 shape。
    print("X shape: ", X.shape)
    print("hidden_linear shape: ", hidden_linear.shape)
    print("hidden shape: ", hidden.shape)
    print("logits shape: ", logits.shape)
    print("probabilities shape: ", probabilities.shape)
    print("\nhidden_linear:\n", hidden_linear)
    print("\nhidden (after ReLU):\n", hidden)
    print("\nlogits:\n", logits)
    print("\nprobabilities:\n", probabilities)
    # 6. 验证每个样本的概率和都接近 1。
    row_sum = np.sum(probabilities, axis=1)
    print("\n每个样本概率之和：", row_sum)
    print("全部接近1？", np.allclose(row_sum, np.ones_like(row_sum)))
    # 7. 使用 np.argmax(probabilities, axis=1) 得到预测类别。
    pred_classes = np.argmax(probabilities, axis=1)
    print("\n预测类别：", pred_classes)



if __name__ == "__main__":
    print("--- 为什么需要非线性 ---")
    demo_why_nonlinearity()

    print("\n--- 练习 1 ---")
    practice_activation_values()

    print("\n--- 练习 2 ---")
    practice_softmax_vector()

    print("\n--- 练习 3 ---")
    practice_softmax_batch()

    print("\n--- 作业 ---")
    homework_tiny_network()
