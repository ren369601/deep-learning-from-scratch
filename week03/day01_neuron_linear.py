"""Week 3 / Day 1: neurons and linear layers with NumPy."""

import numpy as np


def neuron_forward(inputs, weights, bias):
    """Return the linear output of one neuron for one sample."""
    if inputs.ndim != 1 or weights.ndim != 1:
        raise ValueError("inputs 和 weights 必须是一维数组")
    if inputs.shape != weights.shape:
        raise ValueError("inputs 和 weights 的 shape 必须相同")

    return float(inputs @ weights + bias)


def linear_layer_forward(X, weights, bias):
    """Return outputs for a batch passing through multiple neurons."""
    if X.ndim != 2 or weights.ndim != 2:
        raise ValueError("X 和 weights 必须是二维数组")
    if X.shape[1] != weights.shape[0]:
        raise ValueError("输入特征数量和权重矩阵不匹配")
    if bias.shape != (weights.shape[1],):
        raise ValueError("bias 的 shape 必须等于输出特征数量")

    return X @ weights + bias


def demo_one_neuron():
    """Calculate one neuron's output."""
    inputs = np.array([1, 2, 3], dtype=np.float32)
    weights = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    bias = 1.0

    output = neuron_forward(inputs, weights, bias)
    print("inputs:", inputs)
    print("weights:", weights)
    print("output:", output)


def demo_linear_layer():
    """Pass two samples through two neurons."""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=np.float32)
    weights = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ], dtype=np.float32)
    bias = np.array([10, -5], dtype=np.float32)

    output = linear_layer_forward(X, weights, bias)
    print("X shape:", X.shape)
    print("weights shape:", weights.shape)
    print("bias shape:", bias.shape)
    print("output:\n", output)
    print("output shape:", output.shape)


class LinearLayer:
    """A reusable NumPy linear layer."""

    def __init__(self, in_features, out_features, seed=42):
        # TODO:
        # 1. Create rng = np.random.default_rng(seed).
        # 2. Initialize self.weights with shape (in_features, out_features).
        #    Use rng.normal(0.0, 0.1, size=...) and convert to float32.
        # 3. Initialize self.bias as float32 zeros with shape (out_features,).
        rng = np.random.RandomState(seed)
        self.weights = rng.normal(0.0, 0.1, size=(in_features, out_features)).astype(np.float32)
        self.bias = np.zeros(out_features).astype(np.float32)


    def forward(self, X):
        """Return X @ weights + bias after validating shapes."""
        # TODO:
        # 1. Check X is two-dimensional.
        # 2. Check X.shape[1] matches self.weights.shape[0].
        # 3. Return the linear output.
        assert X.ndim==2, "X must be 2D"
        assert X.shape[1]==self.weights.shape[0], "X must have the same number of columns as weights"
        return X @ self.weights + self.bias

def practice_one_neuron():
    """练习 1：计算一个神经元的输出。"""
    inputs = np.array([2, -1, 3], dtype=np.float32)
    weights = np.array([0.5, 2.0, -1.0], dtype=np.float32)
    bias = 1.0

    # TODO:
    # 1. 先手算预期输出。
    # 2. 调用 neuron_forward() 并打印结果。
    # 3. 确认代码结果与手算结果相同。
    output = neuron_forward(inputs, weights, bias)
    print("output:", output)


def practice_multiple_neurons():
    """练习 2：让三个样本通过两个神经元。"""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=np.float32)
    weights = np.array([
        [1, -1],
        [2, 0.5],
        [3, 2],
    ], dtype=np.float32)
    bias = np.array([1, -2], dtype=np.float32)

    # TODO:
    # 1. 写出 X、weights、bias 和 output 的预期 shape。
    # 2. 调用 linear_layer_forward()。
    # 3. 打印 output 和 output.shape。
    # 4. 说明为什么每个样本会得到两个输出。
    output = linear_layer_forward(X, weights, bias)
    print("output:", output)
    print(output.shape)
    print("解释：这一层有2个神经元，每个神经元输出1个数字，所以每个样本输出2个值")


def practice_layer_class():
    """练习 3：创建并使用 LinearLayer 类。"""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=np.float32)

    # TODO:
    # 1. Create LinearLayer(in_features=3, out_features=4, seed=7).
    # 2. Print weights.shape and bias.shape.
    # 3. Call layer.forward(X).
    # 4. Print output and output.shape.
    # Expected output shape: (2, 4).
    layer = LinearLayer(in_features=3, out_features=4, seed=7)
    print("layer.weights.shape =", layer.weights.shape)
    print("layer.bias.shape =", layer.bias.shape)
    out = layer.forward(X)
    print("practice_layer_class output:\n", out)
    print("output shape:", out.shape)


def homework_verify_neurons():
    """作业：实现并验证一个四输入、三输出线性层。"""
    X = np.array([
        [1, 2, 3, 4],
        [2, 0, -1, 3],
        [0, 1, 2, -2],
        [3, 2, 1, 0],
        [-1, 2, 0, 1],
    ], dtype=np.float32)
    weights = np.array([
        [1.0, 0.5, -1.0],
        [2.0, -0.5, 1.5],
        [0.0, 1.0, 2.0],
        [-1.0, 2.0, 0.5],
    ], dtype=np.float32)
    bias = np.array([1.0, -2.0, 3.0], dtype=np.float32)

    # TODO:
    # 1. 写出所有输入和输出的预期 shape。
    # 2. 使用 linear_layer_forward() 得到 output。
    # 3. 打印 output 和 output.shape。
    # 4. 单独计算第一个样本经过第一个神经元的结果：
    #    X[0] @ weights[:, 0] + bias[0]
    # 5. 使用 np.isclose() 验证它等于 output[0, 0]。
    # 6. 解释 weights 的每一列为什么代表一个神经元。
    output = linear_layer_forward(X, weights, bias)
    print("output shape:", output.shape)
    print(output)

    neuron0_sample0 = X[0] @ weights[:, 0] + bias[0]
    print("第一个样本，第一个神经元单独计算结果：", neuron0_sample0)
    print("output[0,0] =", output[0, 0])
    print("是否相等?", np.isclose(neuron0_sample0, output[0, 0]))
    print("解释：weights每一列是一套独立权重，对应一个神经元，做加权求和+bias")



if __name__ == "__main__":
    print("--- 单个神经元示例 ---")
    demo_one_neuron()

    print("\n--- 线性层示例 ---")
    demo_linear_layer()

    print("\n--- 练习 1 ---")
    practice_one_neuron()

    print("\n--- 练习 2 ---")
    practice_multiple_neurons()

    print("\n--- 练习 3 ---")
    practice_layer_class()

    print("\n--- 作业 ---")
    homework_verify_neurons()
