"""Week 4 / Day 3: build and inspect custom nn.Module models."""

import torch
from torch import nn


class DemoLinearModel(nn.Module):
    """A complete one-layer model example."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, X):
        return self.linear(X)


def demo_module():
    """Show model calling, parameter shapes, and parameter count."""
    torch.manual_seed(42)
    model = DemoLinearModel(2, 1)
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output = model(X)

    print("input shape:", X.shape, "output shape:", output.shape)
    for name, parameter in model.named_parameters():
        print(name, parameter.shape)
    print("parameter count:", sum(p.numel() for p in model.parameters()))


class OneLayerClassifier(nn.Module):
    """练习 1：写一个从 3 个特征到 2 个类别分数的模型。"""

    def __init__(self):
        super().__init__()
        # TODO: Create self.linear = nn.Linear(3, 2).
        self.linear = nn.Linear(3,2)

    def forward(self, X):
        # TODO: Return the logits from self.linear(X).
        return self.linear(X)


def practice_model_forward():
    """练习 1：创建模型并检查前向输出。"""
    X = torch.tensor([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 3.0],
        [2.0, 2.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    # TODO: Instantiate OneLayerClassifier, call model(X), and print logits.
    model = OneLayerClassifier()
    logits = model(X)
    # Print X.shape and logits.shape; expect (4, 3) and (4, 2).
    print("X.shape:", X.shape)
    print("logits.shape:", logits.shape)
    print("logits:\n", logits)
    # Explain why the output is called logits rather than probabilities.
    # 解释：这是原始分数，没有经过softmax归一化，不是概率，所以叫logits


def practice_inspect_parameters():
    """练习 2：查看模型自动登记的参数。"""
    model = OneLayerClassifier()

    # TODO: Print each name and shape from model.named_parameters().
    for name, param in model.named_parameters():
        print(name, param.shape)
        # Print total parameter count; expected 8.
    total_params = sum(p.numel() for p in model.parameters())
    print("total params:", total_params)
    # Print list(model.state_dict().keys()).
    print("state_dict keys:", list(model.state_dict().keys()))
    # Compare linear.weight.shape with the NumPy W.shape from Week 3.
    # PyTorch linear.weight: (out_features, in_features) = (2,3)
    # Week3手写numpy版本W: (in_features, out_features) = (3,2)  两者互为转置


class ScaleShift(nn.Module):
    """作业：用自定义参数实现逐特征缩放与平移。"""

    def __init__(self):
        super().__init__()
        # TODO: Register scale = [1.0, 2.0] and bias = [0.0, -1.0]
        # as nn.Parameter attributes.
        self.scale = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.bias = nn.Parameter(torch.tensor([0.0, -1.0]))

    def forward(self, X):
        # TODO: Return X * self.scale + self.bias.
        return X * self.scale + self.bias


def homework_custom_parameters():
    """作业：检查自定义参数、输出和梯度。"""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # TODO: Instantiate ScaleShift and run model(X).
    model = ScaleShift()
    output = model(X)
    # Check output.shape == (3, 2), and hand-calculate the first row.
    print("output:\n", output)
    print("output.shape:", output.shape)
    # 第一行手算：[1*1+0 , 2*2 + (-1)] = [1, 3]
    # Print named_parameters and total parameter count (expected 4).
    for name, param in model.named_parameters():
        print(name, param.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print("total params:", total_params)
    # Set loss = (output ** 2).mean(), call loss.backward().
    loss = (output ** 2).mean()
    loss.backward()
    # Print scale.grad and bias.grad; their shapes should both be (2,).
    print("scale.grad:", model.scale.grad)
    print("bias.grad:", model.bias.grad)


if __name__ == "__main__":
    print("--- nn.Module 示例 ---")
    demo_module()

    print("\n--- 练习 1：定义与调用模型 ---")
    practice_model_forward()

    print("\n--- 练习 2：查看模型参数 ---")
    practice_inspect_parameters()

    print("\n--- 作业：自定义参数 ---")
    homework_custom_parameters()
