"""Week 3 / Day 5: a two-layer NumPy MLP forward pass."""

import numpy as np


def softmax(logits):
    """Apply stable Softmax along the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


class Linear:
    """A fully connected linear layer."""

    def __init__(self, in_features, out_features, seed=42):
        # TODO:
        # 1. Create a default_rng with the given seed.
        rng = np.random.default_rng(seed)
        # 2. Calculate scale = sqrt(2 / in_features).
        scale = np.sqrt(2.0 / in_features)
        # 3. Initialize float32 weights with shape (in_features, out_features).
        self.weights = rng.normal(0.0, scale, size=(in_features, out_features)).astype(np.float32)
        # 4. Initialize float32 bias zeros with shape (out_features,).
        self.bias = np.zeros(out_features, dtype=np.float32)
        # 5. Set self.last_input = None.
        self.last_input = None

    def forward(self, X):
        """Cache X and return X @ weights + bias."""
        # TODO:
        # 1. Validate X is 2D and its feature count matches the weights.
        assert X.ndim == 2, "X must be 2D array"
        assert X.shape[1] == self.weights.shape[0], "Feature dimension mismatch"
        # 2. Save X in self.last_input.
        self.last_input = X
        # 3. Return the linear output.
        return X @ self.weights + self.bias


class ReLU:
    """A ReLU activation layer."""

    def __init__(self):
        # TODO: Set self.last_input = None.
        self.last_input = None

    def forward(self, X):
        """Cache X and apply ReLU."""
        # TODO:
        # 1. Save X in self.last_input.
        # 2. Return np.maximum(0, X).
        self.last_input = X
        return np.maximum(0, X)


class TwoLayerMLP:
    """Linear -> ReLU -> Linear -> Softmax."""

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        # TODO:
        # 1. Create self.linear1 from input_size to hidden_size.
        self.linear1 = Linear(input_size, hidden_size, seed=seed)
        # 2. Create self.relu.
        self.relu = ReLU()
        # 3. Create self.linear2 from hidden_size to output_size.
        #    Use seed + 1 for the second Linear layer.
        self.linear2 = Linear(hidden_size, output_size, seed=seed + 1)
        # 4. Initialize hidden_linear, hidden, logits, probabilities to None.
        self.hidden_linear = None
        self.hidden = None
        self.logits = None
        self.probabilities = None

    def forward(self, X):
        """Run and cache the complete forward pass."""
        # TODO:
        # 1. self.hidden_linear = self.linear1.forward(X)
        self.hidden_linear = self.linear1.forward(X)
        # 2. self.hidden = self.relu.forward(self.hidden_linear)
        self.hidden = self.relu.forward(self.hidden_linear)
        # 3. self.logits = self.linear2.forward(self.hidden)
        self.logits = self.linear2.forward(self.hidden)
        # 4. self.probabilities = softmax(self.logits)
        self.probabilities = softmax(self.logits)
        # 5. Return self.probabilities.
        return self.probabilities

    def parameter_count(self):
        """Return the total number of trainable scalar parameters."""
        # TODO: Sum the sizes of both weight matrices and both bias vectors.
        # Sum the sizes of both weight matrices and both bias vectors.
        w1_num = self.linear1.weights.size
        b1_num = self.linear1.bias.size
        w2_num = self.linear2.weights.size
        b2_num = self.linear2.bias.size
        return w1_num + b1_num + w2_num + b2_num


def practice_linear_layer():
    """练习 1：创建 Linear 层并完成前向传播。"""
    X = np.array([
        [1.0, 2.0],
        [2.0, -1.0],
        [-1.0, 1.0],
        [0.0, 3.0],
    ], dtype=np.float32)

    # TODO:
    # 1. Create Linear(2, 3, seed=7).
    layer = Linear(2, 3, seed=7)
    # 2. Print weights.shape and bias.shape.
    print("weights shape:", layer.weights.shape)
    print("bias shape:", layer.bias.shape)
    # 3. Run forward and print output plus output.shape.
    out = layer.forward(X)
    print("linear output:\n", out)
    print("output shape:", out.shape)
    # 4. Confirm last_input is X or has the same values as X.
    print("last_input equals X?", np.array_equal(layer.last_input, X))



def practice_mlp_shapes():
    """练习 2：追踪两层 MLP 的全部 shape。"""
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=np.float32)

    # TODO:
    # 1. Create TwoLayerMLP(2, 4, 2, seed=42).
    model = TwoLayerMLP(2, 4, 2, seed=42)
    # 2. Run forward.
    model.forward(X)
    # 3. Print X, hidden_linear, hidden, logits, probabilities shapes.
    print("X shape:            ", X.shape)
    print("hidden_linear shape:", model.hidden_linear.shape)
    print("hidden shape:       ", model.hidden.shape)
    print("logits shape:       ", model.logits.shape)
    print("probabilities shape:", model.probabilities.shape)
    # 4. Print model.parameter_count(); expected value is 22.
    print("Total params:", model.parameter_count())  # 预期22
    # 验算：W1(2*4=8)+b1(4) + W2(4*2=8)+b2(2) = 8+4+8+2 = 22



def practice_probabilities():
    """练习 3：验证 Softmax 输出。"""
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=np.float32)

    # TODO:
    # 1. Create and run a TwoLayerMLP(2, 4, 2).
    model = TwoLayerMLP(2, 4, 2)
    model.forward(X)
    prob = model.probabilities
    # 2. Print probabilities and row sums.
    print("probabilities:\n", prob)
    row_sum = np.sum(prob, axis=1)
    print("row sum for each sample:", row_sum)
    # 3. Verify row sums with np.allclose().
    print("All rows sum to 1?", np.allclose(row_sum, 1.0))
    # 4. Print predicted classes using argmax(axis=1).
    pred_class = np.argmax(prob, axis=1)
    print("predicted classes:", pred_class)
    # 5. Explain why predictions are not meaningful before training.
    print("""解释：
    目前权重只是随机He初始化，没有经过梯度下降更新；
    网络没有学到任何数据规律，预测结果只是随机猜测，没有实际分类意义。
    """)


def homework_forward_project():
    """作业：完成一个三输入、五隐藏、三分类的 MLP Forward。"""
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 0.0, -1.0],
        [-1.0, 1.0, 2.0],
        [0.0, 3.0, -2.0],
        [2.0, -1.0, 1.0],
    ], dtype=np.float32)

    # TODO:
    # 1. Create TwoLayerMLP(input_size=3, hidden_size=5, output_size=3).
    model = TwoLayerMLP(input_size=3, hidden_size=5, output_size=3)
    # 2. Before running, write all expected shapes in comments.
    """
    X: (5, 3)    5个样本，3个输入特征
    hidden_linear (linear1输出): (5, 5)
    hidden (relu输出): (5,5)
    logits (linear2输出): (5,3)
    probabilities (softmax): (5,3)
    参数计算：W1(3*5=15)+b1(5) + W2(5*3=15)+b2(3) =15+5+15+3 =38
    """
    # 3. Run forward and print every intermediate array and shape.
    model.forward(X)
    print("X shape:            ", X.shape)
    print("hidden_linear:\n", model.hidden_linear)
    print("hidden_linear shape:", model.hidden_linear.shape)
    print("hidden:\n", model.hidden)
    print("hidden shape:       ", model.hidden.shape)
    print("logits:\n", model.logits)
    print("logits shape:       ", model.logits.shape)
    print("probabilities:\n", model.probabilities)
    print("probabilities shape:", model.probabilities.shape)
    # 4. Print parameter count; expected value is 38.
    print("Total parameters:", model.parameter_count())  # 预期38
    # 5. Verify every probability row sums to 1.
    row_sum = np.sum(model.probabilities, axis=1)
    print("Each row sum:", row_sum)
    print("All sum to 1?", np.allclose(row_sum, 1.0))
    # 6. Print predicted classes.
    pred_classes = np.argmax(model.probabilities, axis=1)
    print("predicted classes: ", pred_classes)
    # 7. Explain the meaning of one row in hidden and probabilities.
    print("""解释：
    hidden 一行：对应一个样本，经过第一层线性+ReLU之后，5个隐藏神经元的激活值，是学习到的中间特征表示。
    probabilities 一行：对应一个样本，3个类别的预测概率，三个概率相加=1，代表模型认为该样本属于每一类的可能性。
    """)


if __name__ == "__main__":
    print("--- 练习 1：Linear ---")
    practice_linear_layer()

    print("\n--- 练习 2：MLP Shapes ---")
    practice_mlp_shapes()

    print("\n--- 练习 3：Probabilities ---")
    practice_probabilities()

    print("\n--- 作业：完整 Forward ---")
    homework_forward_project()
