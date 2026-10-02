"""Week 2 / Day 7: integrated NumPy and gradient-descent review."""

import numpy as np


def linear_forward(X, weights, bias):
    """Return batch predictions for a multi-feature linear model."""
    # TODO:
    # 1. Check that X is two-dimensional.
    # 2. Check that X.shape[1] matches weights.shape[0].
    # 3. Return X @ weights + bias.
    assert X.ndim == 2, "X must be 2D array"
    assert X.shape[1] == weights.shape[0], "feature dimension mismatch"
    return X @ weights + bias


def mean_squared_error(predictions, targets):
    """Return MSE after validating both shapes."""
    # TODO:
    # 1. Check predictions.shape == targets.shape.
    # 2. Return MSE as a Python float.
    assert predictions.shape==targets.shape, "predictions and targets must have the same shape"
    error=predictions - targets
    error_squared = error**2
    mse=np.mean(error_squared)

    return float(mse)

def compute_gradients(X, targets, weights, bias):
    """Return gradients for the weight vector and scalar bias."""
    # TODO:
    # 1. Calculate predictions and errors.
    # 2. Calculate grad_weights using X.T @ errors.
    # 3. Calculate grad_bias.
    # 4. Return both gradients.
    predictions = linear_forward(X, weights, bias)
    errors= predictions - targets
    n = X.shape[0]
    grad_weights=(2/n)*(X.T @ errors)
    grad_bias=2*errors.mean()
    return grad_weights, grad_bias


def train(X, targets, learning_rate, epochs):
    """Train from zero parameters and return parameters plus loss history."""
    # TODO:
    # 1. Initialize weights from X.shape[1], and initialize bias.
    # 2. Create an empty loss_history.
    # 3. For every epoch: forward, loss, gradients, parameter update.
    # 4. Append each loss to loss_history.
    # 5. Return weights, bias, loss_history.
    n_features = X.shape[1]
    weights = np.zeros(n_features, dtype=np.float32)
    bias = 0.0
    loss_history = []

    for _ in range(epochs):
        pred = linear_forward(X, weights, bias)
        loss = mean_squared_error(pred, targets)
        loss_history.append(loss)
        gw, gb = compute_gradients(X, targets, weights, bias)
        weights -= learning_rate * gw
        bias -= learning_rate * gb
    return weights, bias, loss_history


def shape_quiz():
    """练习 1：运行前先在注释中写出每一步的预期 shape。"""
    values = np.arange(24)

    # TODO:
    # 1. Reshape values to (2, 3, 4).
    # 2. Transpose it with axes (0, 2, 1).
    # 3. Add a broadcast value with shape (1, 4, 1).
    # 4. Print every shape.
    #
    # Before running, write the expected final shape in a comment.
    arr = values.reshape(2, 3, 4)
    print("after reshape:", arr.shape)
    arr_t = arr.transpose(0, 2, 1)
    print("after transpose:", arr_t.shape)
    add_term = np.ones((1, 4, 1))
    res = arr_t + add_term
    print("after broadcast add:", res.shape)



def matrix_quiz():
    """练习 2：分析并执行一个多输出线性计算。"""
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    weights = np.arange(6, dtype=np.float32).reshape(3, 2)
    bias = np.array([10, -5], dtype=np.float32)

    # TODO:
    # 1. Write the expected output shape before running.
    # 2. Calculate output = X @ weights + bias.
    # 3. Print every input shape, output, and output.shape.
    print("X shape:", X.shape)
    print("weights shape:", weights.shape)
    print("bias shape:", bias.shape)
    output = X @ weights + bias
    print("output:\n", output)
    print("output shape:", output.shape)


def homework_train_model():
    """综合作业：从头训练三个特征的线性模型。"""
    X = np.array([
        [-2, -1, 0],
        [-1, 0, 1],
        [0, 1, 2],
        [1, 2, -1],
        [2, -2, 1],
        [-1, 2, 2],
        [2, 1, -2],
        [0, -2, -1],
    ], dtype=np.float32)
    true_weights = np.array([2.0, -1.5, 0.5], dtype=np.float32)
    true_bias = 3.0
    targets = X @ true_weights + true_bias

    # TODO:
    # 1. Call train() with a suitable learning rate and enough epochs.
    # 2. Print first loss and final loss.
    # 3. Print learned weights, bias, predictions, and targets.
    # 4. Confirm final loss is smaller than first loss.
    # 5. Confirm learned parameters are close to true parameters.
    #
    # Do not copy the Day 6 class. Complete the functions in this file.
    lr = 0.05
    epochs = 8000
    w, b, loss_hist = train(X, targets, lr, epochs)

    print(loss_hist[0])
    print(loss_hist[-1])
    print(w, b)
    pred=linear_forward(X, w, b)
    print(pred)
    print(targets)

    print("Final loss < first loss ?", loss_hist[-1] < loss_hist[0])


if __name__ == "__main__":
    print("--- Shape 综合题 ---")
    shape_quiz()

    print("\n--- 矩阵乘法综合题 ---")
    matrix_quiz()

    print("\n--- 梯度下降综合作业 ---")
    homework_train_model()
