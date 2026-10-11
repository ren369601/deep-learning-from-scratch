"""Week 4 / Day 5: assemble a two-layer PyTorch MLP for XOR."""

import torch
from torch import nn


def demo_sequential_mlp():
    """A complete example of Linear -> ReLU -> Linear."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    logits = model(X)
    print("X shape:", X.shape, "logits shape:", logits.shape)
    print("logits:\n", logits)


class TwoLayerMLP(nn.Module):
    """练习 1：自己定义可复用的两层 MLP。"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # TODO: Register linear1, relu, and linear2 as model attributes.
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # TODO: X -> linear1 -> ReLU -> linear2; return logits.
        hidden = self.relu(self.linear1(X))
        logits = self.linear2(hidden)
        return logits


def practice_forward_shapes():
    """练习 1：检查自定义模型的前向输出。"""
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    # TODO: Set torch.manual_seed(42), create TwoLayerMLP(2, 8, 2).
    # Call model(X), print X.shape, logits.shape, and logits.
    # Check: logits.shape == (4, 2).
    torch.manual_seed(42)
    model = TwoLayerMLP(2, 8, 2)
    logits = model(X)
    # Call model(X), print X.shape, logits.shape, and logits.
    print("X.shape:", X.shape)
    print("logits.shape:", logits.shape)
    print("logits:\n", logits)
    # Check: logits.shape == (4, 2).
    assert logits.shape == (4, 2)


def practice_parameter_shapes():
    """练习 2：列出所有可学习参数。"""
    model = TwoLayerMLP(2, 8, 2)

    # TODO: Print each name and shape from model.named_parameters().
    # Calculate and print the total parameter count; expected 42.
    # Explain why linear1.weight has shape (8, 2), not (2, 8).

    for name, param in model.named_parameters():
        print(name, param.shape)
        # Calculate and print the total parameter count; expected 42.
    total_params = sum(p.numel() for p in model.parameters())
    print("total params:", total_params)
    # Explain why linear1.weight has shape (8, 2), not (2, 8).
    # 解释：PyTorch nn.Linear权重格式 (out_features, in_features)
    # linear1：输入2，输出8，所以weight shape=(8,2)


def homework_xor_forward_backward():
    """作业：在 XOR 上做一次前向、损失和反向传播。"""
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    torch.manual_seed(42)
    model = TwoLayerMLP(2, 8, 2)

    # TODO:
    # 1. Get logits = model(X); print logits and their shape.
    # 2. Compute CrossEntropyLoss directly from logits and targets.
    # 3. For display only: compute probabilities with softmax(dim=1)
    #    and predicted classes with logits.argmax(dim=1).
    # 4. Call loss.backward(); print parameter and gradient shapes.
    # 5. Check each gradient is not None and has the parameter's shape.
    # 6. Explain why there is no Softmax in TwoLayerMLP.forward().
    # Do not update parameters today; training starts on Day 6.

    # 1. Get logits = model(X); print logits and their shape.
    logits = model(X)
    print("logits shape:", logits.shape)
    print("logits:\n", logits)

    # 2. Compute CrossEntropyLoss directly from logits and targets.
    loss_fn = nn.CrossEntropyLoss()   # 算完前向传播
    loss = loss_fn(logits, targets)
    print("loss:", loss.item())

    # 3. For display only: compute probabilities with softmax(dim=1)
    #    and predicted classes with logits.argmax(dim=1).
    probs = torch.softmax(logits, dim=1)
    pred_classes = logits.argmax(dim=1)
    print("probabilities:\n", probs)
    print("predicted classes:", pred_classes)

    # 4. Call loss.backward(); print parameter and gradient shapes.
    loss.backward()
    print("\n--- 参数与梯度 ---")
    for name, param in model.named_parameters():
        print(name, "param shape:", param.shape, "grad shape:", param.grad.shape)

    # 5. Check each gradient is not None and has the parameter's shape.
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert param.grad.shape == param.shape

    # 6. Explain why there is no Softmax in TwoLayerMLP.forward().
    # 原因：nn.CrossEntropyLoss内部自带softmax计算，forward直接返回logits即可，
    # 如果手动加softmax会重复计算，数值不稳定，梯度计算出错。

    # Do not update parameters today; training starts on Day 6.


if __name__ == "__main__":
    print("--- 两层 MLP 示例 ---")
    demo_sequential_mlp()

    print("\n--- 练习 1：自定义 MLP 前向 ---")
    practice_forward_shapes()

    print("\n--- 练习 2：参数 shape 与数量 ---")
    practice_parameter_shapes()

    print("\n--- 作业：XOR 前向与反向 ---")
    homework_xor_forward_backward()
