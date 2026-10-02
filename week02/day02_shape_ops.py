"""Week 2 / Day 2: reshape, transpose, and broadcasting."""

import numpy as np


def demo_reshape():
    """Reshape twelve values without changing their total count."""
    numbers = np.arange(12)
    matrix = numbers.reshape(3, 4)
    tensor = numbers.reshape(2, 2, 3)
    flat = matrix.reshape(-1)

    print("numbers:", numbers, numbers.shape)
    print("matrix:\n", matrix, "shape:", matrix.shape)
    print("tensor:\n", tensor, "shape:", tensor.shape)
    print("flat:", flat, flat.shape)


def demo_transpose():
    """Transpose a two-dimensional matrix."""
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])
    transposed = matrix.T

    print("matrix:\n", matrix, "shape:", matrix.shape)
    print("transposed:\n", transposed, "shape:", transposed.shape)


def demo_broadcasting():
    """Broadcast a row vector and a column vector."""
    samples = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])
    feature_bias = np.array([10, 20, 30])
    row_bonus = np.array([
        [100],
        [200],
    ])

    print("按特征广播:\n", samples + feature_bias)
    print("按样本广播:\n", samples + row_bonus)


def practice_reshape():
    """练习 1：改变数组的 shape。"""
    numbers = np.arange(24)

    # TODO:
    # 1. 将 numbers reshape 为 (4, 6)，打印数组和 shape。
    # 2. 将 numbers reshape 为 (2, 3, 4)，打印数组和 shape。
    # 3. 使用 -1 将三维结果重新展平，打印数组和 shape。
    # 4. 思考三个数组的 size 是否相同。
    matrix=numbers.reshape(4,6)
    print(matrix)
    print(matrix.shape)
    tensor=numbers.reshape(2,3,4)
    print(tensor)
    print(tensor.shape)
    flat=tensor.reshape(-1)
    print(flat)
    print(flat.shape)



def practice_transpose():
    """练习 2：转置二维和三维数据。"""
    matrix = np.arange(6).reshape(2, 3)
    tensor = np.zeros((2, 3, 4))

    # TODO:
    # 1. 使用 matrix.T 得到 shape 为 (3, 2) 的结果并打印。
    # 2. 使用 np.transpose(tensor, (0, 2, 1)) 得到新 tensor。
    # 3. 打印原 tensor 和新 tensor 的 shape。
    matrix1=matrix.T
    print(matrix1)
    print(matrix1.shape)
    tensor1=np.transpose(tensor,(0,2,1))
    print(tensor.shape)
    print(tensor1.shape)



def practice_broadcasting():
    """练习 3：分别进行按列和按行广播。"""
    samples = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
    ])
    feature_adjustment = np.array([1, 2, 3])
    sample_bonus = np.array([
        [100],
        [200],
        [300],
    ])

    # TODO:
    # 1. 计算 samples + feature_adjustment 并打印结果和 shape。
    # 2. 计算 samples + sample_bonus 并打印结果和 shape。
    # 3. 用自己的话解释两个广播结果为什么不同。
    matrix=samples+feature_adjustment
    print(matrix)
    print(matrix.shape)

    tensor=samples+sample_bonus
    print(tensor)
    print(tensor.shape)



def homework_shape_pipeline():
    """作业：完成一次 reshape、transpose 和 broadcasting 流程。"""
    # TODO:
    # 1. 用 np.arange(12) 创建一维数组。
    # 2. reshape 为 (3, 4)，把它当作 3 个样本、每个样本 4 个特征。
    # 3. 创建 feature_bias = np.array([10, 20, 30, 40])。
    # 4. 将 feature_bias 广播加到每个样本上。
    # 5. 把结果转置为 shape (4, 3)。
    # 6. 依次打印每一步的内容和 shape。
    matrix=np.arange(12)
    print(matrix)
    print(matrix.shape)
    tensor=matrix.reshape(3,4)
    print(tensor)
    print(tensor.shape)
    feature_bias=np.array([10, 20, 30, 40])
    print(feature_bias)
    print(feature_bias.shape)
    mat=tensor+feature_bias
    print(mat)
    print(mat.shape)
    mat1=mat.T
    print(mat1)
    print(mat1.shape)



if __name__ == "__main__":
    print("--- Reshape 示例 ---")
    demo_reshape()

    print("\n--- Transpose 示例 ---")
    demo_transpose()

    print("\n--- Broadcasting 示例 ---")
    demo_broadcasting()

    print("\n--- 练习 1 ---")
    practice_reshape()

    print("\n--- 练习 2 ---")
    practice_transpose()

    print("\n--- 练习 3 ---")
    practice_broadcasting()

    print("\n--- 作业 ---")
    homework_shape_pipeline()
