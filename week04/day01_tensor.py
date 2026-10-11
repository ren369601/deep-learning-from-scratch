"""Week 4 / Day 1: PyTorch Tensor, shapes, and basic operations."""

import torch


def demo_tensor_basics():
    """A complete example to run before the exercises."""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    W = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    b = torch.tensor([10.0, 20.0], dtype=torch.float32)

    Z = X @ W + b
    print("X:", X)
    print("X shape / dtype / device:", X.shape, X.dtype, X.device)
    print("Z = X @ W + b:\n", Z)
    print("Z shape:", Z.shape)


def practice_create_tensors():
    """Practice 1: create training data and integer class labels."""
    # TODO: Create X with shape (3, 2) and dtype torch.float32.
    # Rows: [1, 2], [3, 4], [5, 6].
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    # TODO: Create targets = [0, 1, 0] with dtype torch.long.
    targets = torch.tensor([0, 1, 0], dtype=torch.long)
    print("X:\n", X)
    print("X shape / dtype:", X.shape, X.dtype)
    print("targets:", targets)
    print("targets shape / dtype:", targets.shape, targets.dtype)
    assert X.shape == (3, 2) and X.dtype == torch.float32
    assert targets.shape == (3,) and targets.dtype == torch.long


def practice_linear_layer():
    """Practice 2: calculate the output of a linear layer."""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([10.0, -5.0])

    # TODO: Compute output = X @ W + b and print it and its shape.
    output = X @ W + b
    print("output:\n", output)
    print("output shape:", output.shape)
    # Before running, calculate the first row by hand: [?, ?].
    # Check: output.shape == (3, 2).
    assert output.shape == (3, 2)
    assert torch.equal(output[0], torch.tensor([17.0, 5.0]))


def homework_reshape_and_mean():
    """Homework: reshape, transpose, and center each column."""
    numbers = torch.arange(6, dtype=torch.float32)

    # TODO: Reshape numbers to matrix with shape (2, 3).
    matrix = numbers.reshape(2, 3)
    # TODO: Transpose matrix and print its shape and values.
    matrix_T = matrix.T
    print("matrix transposed:\n", matrix_T)
    print("matrix_T shape:", matrix_T.shape)
    # TODO: Calculate each column's mean with keepdim=True.
    col_means = matrix.mean(dim=0, keepdim=True)
    print("column means:\n", col_means)
    # TODO: Subtract the column means from matrix and print the result.
    centered = matrix - col_means
    print("centered matrix:\n", centered)
    # Check: centered.shape == (2, 3); centered.mean(dim=0) is near zero.
    assert centered.shape == (2, 3)
    assert torch.allclose(centered.mean(dim=0), torch.zeros(3))


if __name__ == "__main__":
    print("--- Tensor 基础示例 ---")
    demo_tensor_basics()
    print("\n--- 练习 1：创建 Tensor ---")
    practice_create_tensors()
    print("\n--- 练习 2：线性层计算 ---")
    practice_linear_layer()
    print("\n--- 作业：reshape、转置与列平均 ---")
    homework_reshape_and_mean()
