"""Week 2 / Day 1: NumPy arrays, shape, ndim, and dtype."""

import numpy as np


def describe_array(name, array):
    """Print the most important properties of a NumPy array."""
    print(name)
    print(array)
    print("shape:", array.shape)
    print("ndim:", array.ndim)
    print("dtype:", array.dtype)
    print("size:", array.size)


def demo_vector_and_matrix():
    """Show the difference between a vector and a matrix."""
    vector = np.array([1, 2, 3], dtype=np.float32)
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=np.float32)

    describe_array("vector", vector)
    print()
    describe_array("matrix", matrix)


def demo_array_creation():
    """Create arrays with zeros, ones, and arange."""
    zeros = np.zeros((2, 3))
    ones = np.ones((2, 3))
    numbers = np.arange(0, 6)

    print("zeros:\n", zeros)
    print("ones:\n", ones)
    print("numbers:", numbers)


def practice_array_properties():
    """练习 1：创建数组并查看它的属性。"""
    # TODO:
    # 1. 创建 vector = np.array([2, 4, 6, 8], dtype=np.float32)。
    # 2. 打印 vector、shape、ndim、dtype 和 size。
    # 3. 打印第一个元素和最后一个元素。
    vector=np.array([2,4,6,8], dtype=np.float32)
    describe_array("vector", vector)
    print(vector[0])
    print(vector[-1])


def practice_matrix_indexing():
    """练习 2：练习二维 array 的索引。"""
    matrix = np.array([
        [10, 20, 30],
        [40, 50, 60],
    ])

    # TODO:
    # 1. 打印 matrix 的 shape。
    # 2. 打印第一行。
    # 3. 打印第二行第三列的元素 60。
    # 4. 使用 matrix[:, 1] 打印所有行的第二列。
    print(matrix.shape)
    print(matrix[0])
    print(matrix[1,2])
    print(matrix[:,1])

def homework_student_dataset():
    """作业：用二维 array 表示一批学生数据。"""
    # 每一行代表一个学生。
    # 四列依次表示：学习小时、睡眠小时、作业数、缺课数。
    #
    # TODO:
    # 1. 创建至少 3 个学生的数据，指定 dtype=np.float32。
    student_array = np.array([
        [8.0, 7.0, 5.0, 1.0],  # 学生1
        [6.0, 6.5, 3.0, 0.0],  # 学生2
        [9.0, 5.0, 4.0, 2.0],  # 学生3
    ], dtype=np.float32)
    # 2. 打印整个数据集。
    print("学生数据集：")
    print(student_array)
    # 3. 打印 shape、ndim、dtype 和 size。
    print("shape:", student_array.shape)
    print("ndim:", student_array.ndim)
    print("dtype:", student_array.dtype)
    print("size:", student_array.size)
    # 4. 打印第二个学生的全部特征。（下标1）
    print("第二个学生：", student_array[1])
    # 5. 使用二维索引打印所有学生的第三个特征（作业数，下标2）。
    print("所有学生作业数：", student_array[:, 2])
    # 6. 将整个数据集乘以 2，并打印结果；观察 shape 是否改变。
    student_times2 = student_array * 2
    print("全部数据*2：")
    print(student_times2)
    print("乘2之后shape：", student_times2.shape)


if __name__ == "__main__":
    print("--- 向量与矩阵 ---")
    demo_vector_and_matrix()

    print("\n--- 创建数组 ---")
    demo_array_creation()

    print("\n--- 练习 1 ---")
    practice_array_properties()

    print("\n--- 练习 2 ---")
    practice_matrix_indexing()

    print("\n--- 作业 ---")
    homework_student_dataset()
