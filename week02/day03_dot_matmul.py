"""Week 2 / Day 3: dot products and matrix multiplication."""

import numpy as np


def manual_dot(a, b):
    """Calculate a dot product using a Python loop."""
    if len(a) != len(b):
        raise ValueError("a 和 b 的长度必须相同")

    total = 0
    for value_a, value_b in zip(a, b):
        total += value_a * value_b

    return total


def demo_dot_product():
    """Compare element-wise multiplication and dot product."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    print("a * b:", a * b)
    print("manual dot:", manual_dot(a, b))
    print("a @ b:", a @ b)
    print("np.dot(a, b):", np.dot(a, b))


def demo_batch_linear():
    """Use matrix-vector multiplication for batch prediction."""
    samples = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ])
    weights = np.array([10, 20])
    bias = 5

    predictions = samples @ weights + bias

    print("samples shape:", samples.shape)
    print("weights shape:", weights.shape)
    print("predictions:", predictions)
    print("predictions shape:", predictions.shape)


def practice_dot_product():
    """练习 1：用三种方法计算同一个点积。"""
    a = np.array([2, 4, 6])
    b = np.array([1, 3, 5])

    # TODO:
    # 1. 使用 manual_dot(a, b) 计算并打印结果。
    # 2. 使用 a @ b 计算并打印结果。
    # 3. 使用 (a * b).sum() 计算并打印结果。
    # 4. 确认三个结果是否相同。
    print(manual_dot(a, b))
    result=a@b
    print(result)
    res=(a*b).sum()
    print(res)
    if(manual_dot(a,b)==result==res):
        print("same")

def practice_matrix_multiplication():
    """练习 2：完成二维矩阵乘法。"""
    matrix_a = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])
    matrix_b = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ])

    # TODO:
    # 1. 先写出 matrix_a 和 matrix_b 的 shape。
    # 2. 判断结果 shape 应该是什么。
    # 3. 计算 result = matrix_a @ matrix_b。
    # 4. 打印 result 和 result.shape。
    print(matrix_a.shape)
    print(matrix_b.shape)
    result=matrix_a @ matrix_b
    print(result)
    print(result.shape)


def practice_batch_prediction():
    """练习 3：用矩阵乘法重写批量线性预测。"""
    samples = np.array([
        [2, 5, 0],
        [4, 4, 1],
        [1, 2, 3],
    ], dtype=np.float32)
    weights = np.array([5, 3, -4], dtype=np.float32)
    bias = 40

    # TODO:
    # 1. 用 samples @ weights + bias 一次计算三个预测。
    # 2. 打印 samples、weights 和 predictions 的 shape。
    # 3. 打印 predictions。
    # 预期结果与 Week 1 相同：[65, 68, 39]。
    predictions= samples @ weights + bias
    print(samples.shape)
    print(weights.shape)
    print(predictions.shape)
    print(predictions)



def homework_multi_output_layer():
    """作业：实现一个有三个输入、两个输出的线性层。"""
    inputs = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ], dtype=np.float32)
    weights = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ], dtype=np.float32)
    bias = np.array([10, -5], dtype=np.float32)

    # TODO:
    # 1. 不运行代码，先写出 inputs、weights、bias 和 output 的预期 shape。
    # 2. 计算 output = inputs @ weights + bias。
    # 3. 打印所有输入 shape 和 output.shape。
    # 4. 打印 output。
    # 5. 解释为什么 4 个样本最终各自得到 2 个输出。
    print(inputs.shape)
    print(weights.shape)
    print(bias.shape)
    print("(4,2)")
    output=inputs @ weights + bias
    print(output)
    print(output.shape)


if __name__ == "__main__":
    print("--- 点积示例 ---")
    demo_dot_product()

    print("\n--- 批量线性预测示例 ---")
    demo_batch_linear()

    print("\n--- 练习 1 ---")
    practice_dot_product()

    print("\n--- 练习 2 ---")
    practice_matrix_multiplication()

    print("\n--- 练习 3 ---")
    practice_batch_prediction()

    print("\n--- 作业 ---")
    homework_multi_output_layer()
