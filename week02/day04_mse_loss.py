"""Week 2 / Day 4: vectorized mean squared error with NumPy."""

import numpy as np


def mean_squared_error(predictions, targets):
    """Return the MSE between arrays with the same shape."""
    if predictions.shape != targets.shape:
        raise ValueError("predictions 和 targets 的 shape 必须相同")

    squared_errors = (predictions - targets) ** 2
    return float(squared_errors.mean())


def demo_mse_steps():
    """Show every step of a vectorized MSE calculation."""
    predictions = np.array([2.5, 5.0, 8.5, 10.0])
    targets = np.array([3.0, 5.0, 7.0, 9.0])

    errors = predictions - targets
    squared_errors = errors ** 2
    loss = squared_errors.mean()

    print("predictions:", predictions)
    print("targets:", targets)
    print("errors:", errors)
    print("squared errors:", squared_errors)
    print("MSE:", loss)


def demo_axis():
    """Compare overall, per-output, and per-sample means."""
    squared_errors = np.array([
        [1.0, 4.0],
        [0.0, 1.0],
        [9.0, 0.0],
    ])

    print("overall mean:", squared_errors.mean())
    print("per-output mean:", squared_errors.mean(axis=0))
    print("per-sample mean:", squared_errors.mean(axis=1))


def practice_mse_steps():
    """练习 1：完整写出 NumPy MSE 的每一步。"""
    predictions = np.array([4.0, 6.0, 8.0, 10.0])
    targets = np.array([3.0, 5.0, 7.0, 9.0])

    # TODO:
    # 1. 计算并打印 errors。
    # 2. 计算并打印 squared_errors。
    # 3. 使用 mean() 计算并打印 MSE。
    # 4. 调用 mean_squared_error()，确认结果相同。
    errors=predictions - targets
    print(errors)
    squared_errors = errors ** 2
    print(squared_errors)
    MSE=squared_errors.mean()
    print(MSE)
    ans=mean_squared_error(predictions, targets)
    if MSE==ans:
        print("same")



def practice_compare_models():
    """练习 2：比较两个线性模型的 MSE。"""
    inputs = np.array([
        [1, 2],
        [2, 0],
        [0, 3],
        [4, 1],
    ], dtype=np.float32)
    targets = np.array([3, 7, 0, 10], dtype=np.float32)

    weights_a = np.array([1, 1], dtype=np.float32)
    bias_a = 0

    weights_b = np.array([2, -1], dtype=np.float32)
    bias_b = 3

    # TODO:
    # 1. 分别用 inputs @ weights + bias 计算两个模型的 predictions。
    # 2. 调用 mean_squared_error() 计算两个 loss。
    # 3. 打印 predictions_a、predictions_b、loss_a、loss_b。
    # 4. 判断哪个模型更好，并解释原因。
    pre1=inputs@weights_a+bias_a
    pre2=inputs@weights_b+bias_b

    loss1=mean_squared_error(pre1, targets)
    loss2=mean_squared_error(pre2, targets)
    print(pre1)
    print(pre2)
    print(loss1)
    print(loss2)

    if loss1<loss2:
        print("1 is better")
    elif loss1==loss2:
        print("same")
    else:
        print("2 is better")



def practice_multi_output_mse():
    """练习 3：计算多输出模型的不同 MSE。"""
    predictions = np.array([
        [2.0, 5.0],
        [4.0, 8.0],
        [6.0, 9.0],
    ])
    targets = np.array([
        [3.0, 5.0],
        [4.0, 7.0],
        [5.0, 10.0],
    ])

    # TODO:
    # 1. 计算 squared_errors。
    # 2. 计算所有元素的 overall_mse。
    # 3. 使用 axis=0 计算 per_output_mse。
    # 4. 使用 axis=1 计算 per_sample_mse。
    # 5. 打印所有结果和它们的 shape。
    errors=predictions - targets
    squared_errors = errors ** 2
    overall_mse=squared_errors.mean()
    per_output_mse=squared_errors.mean(axis=0)
    per_sample_mse=squared_errors.mean(axis=1)

    print("squared_errors:\n", squared_errors)
    print("squared_errors.shape =", squared_errors.shape)
    print("overall_mse =", overall_mse)
    print("per_output_mse =", per_output_mse)
    print("per_output_mse.shape =", per_output_mse.shape)
    print("per_sample_mse =", per_sample_mse)
    print("per_sample_mse.shape =", per_sample_mse.shape)


def homework_evaluate_linear_layer():
    """作业：完成线性层前向计算并评估损失。"""
    inputs = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=np.float32)
    weights = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ], dtype=np.float32)
    bias = np.array([10, -5], dtype=np.float32)
    targets = np.array([
        [30, 25],
        [60, 60],
        [85, 100],
    ], dtype=np.float32)

    # TODO:
    # 1. 计算 predictions = inputs @ weights + bias。
    # 2. 检查 predictions 和 targets 的 shape。
    # 3. 计算 overall MSE。
    # 4. 计算每个输出的 MSE，使用 axis=0。
    # 5. 打印 predictions、targets 和所有 loss。
    # 6. 找出两个输出中哪一个误差更大。
    predictions=inputs@weights + bias
    print(predictions)
    print(targets)
    print(predictions.shape)
    print(targets.shape)

    squared_errors = ((predictions - targets)**2)
    overall_MSE=squared_errors.mean()
    print(squared_errors)
    print(overall_MSE)

    per_output_mse = squared_errors.mean(axis=0)
    print(per_output_mse)

    if per_output_mse[0] > per_output_mse[1]:
        print("\n第0个输出误差更大")
    else:
        print("\n第1个输出误差更大")



if __name__ == "__main__":
    print("--- MSE 分步示例 ---")
    demo_mse_steps()

    print("\n--- Axis 示例 ---")
    demo_axis()

    print("\n--- 练习 1 ---")
    practice_mse_steps()

    print("\n--- 练习 2 ---")
    practice_compare_models()

    print("\n--- 练习 3 ---")
    practice_multi_output_mse()

    print("\n--- 作业 ---")
    homework_evaluate_linear_layer()
