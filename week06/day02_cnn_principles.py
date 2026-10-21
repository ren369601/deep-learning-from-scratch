"""Week 6 / Day 2: channels, shared weights, translation, and pooling."""

import torch
from torch import nn
from torch.nn import functional as F

from day01_convolution import manual_corr2d


def demo_channel_contributions():
    """Example: two input channels contribute to one output value."""
    first = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
    image = torch.stack([first, 10 * first])
    kernels = torch.tensor([[[1.0, 0.0], [0.0, -1.0]],
                            [[0.0, 1.0], [1.0, 0.0]]])
    contributions = (image[:, :2, :2] * kernels).sum(dim=(1, 2))
    print("input shape:", image.shape)
    print("per-input-channel contributions:", contributions)
    print("one output value with bias=1:", contributions.sum().item() + 1)


def shift_right(values):
    """Shift one spatial column right, with zero fill and no wraparound."""
    shifted = torch.zeros_like(values)
    shifted[..., 1:] = values[..., :-1]
    return shifted


def demo_translation_and_pooling():
    """Example: a translation away from boundaries and a pooling counterexample."""
    image = torch.zeros((1, 1, 7, 7))
    image[0, 0, 2, 2] = 1
    kernel = torch.tensor([[[[1.0, 0.0, -1.0],
                              [1.0, 0.0, -1.0],
                              [1.0, 0.0, -1.0]]]])
    original = F.conv2d(image, kernel, padding=1)
    moved = F.conv2d(shift_right(image), kernel, padding=1)
    print("response shifts with the interior point:",
          torch.allclose(moved, shift_right(original)))
    # This point and its response stay away from the boundary.
    # This equality is not a claim for every boundary, input, or stride.
    point = torch.zeros((1, 1, 2, 4))
    point[0, 0, 0, 0] = 1
    for shift in range(3):
        print("point shift:", shift, "pooled:", F.max_pool2d(point, 2, stride=2)[0, 0])
        point = shift_right(point)


def practice_parameter_counts():
    """Practice 1: distinguish output elements from registered parameters."""
    X = torch.zeros((2, 3, 8, 8))  # 2张图片，每张3个通道，高宽8
    layer = nn.Conv2d(3, 4, kernel_size=3, padding=1) # 卷积层：输入通道3，输出通道4，卷积核3*3，pedding=1

    # TODO:
    # 1. Predict then print input, weight, bias, and output shapes.
    # 2. Count parameters using both the formula and model.parameters().
    # 3. Create Linear(3*8*8, 4*8*8) for the same per-image output element count.
    # 4. Print both parameter counts; explain local connections and sharing.
    # 5. Explain why neither parameter count is multiplied by batch size.
    print("Practice 1: complete parameter counts and shape checks.")
    # 1. shapes
    print("X input shape:", X.shape)
    print("Conv weight shape:", layer.weight.shape)
    print("Conv bias shape:", layer.bias.shape)
    out = layer(X)
    print("Conv output shape:", out.shape)

    # 2. count parameters
    weight_params = layer.weight.numel()
    bias_params = layer.bias.numel()
    total_params = weight_params + bias_params
    print(f"weight params: {weight_params}, bias params: {bias_params}, total: {total_params}")
    # formula: Cout * Cin * Kh * Kw + Cout
    formula_total = 4 * 3 * 3 * 3 + 4
    print("formula total params:", formula_total)

    # 3. Linear layer with same per-image output elements
    # Conv output spatial size: H=8,W=8, Cout=4 → per image elements:8*8*4 = 256
    # 输入每张图 3*8*8=192 项，输出每张图 4*8*8=256 项。
    lin = nn.Linear(3 * 8 * 8, 4 * 8 * 8)
    lin_total = lin.weight.numel() + lin.bias.numel()
    print("Linear layer total params:", lin_total)
    assert total_params == formula_total == 112
    assert lin_total == 49408
    assert out.shape == (2, 4, 8, 8)

    # 4. compare
    print("Conv uses local connectivity + weight sharing, so much fewer params than Linear.")
    # 5. batch size does NOT affect parameter count. Parameters are kernel weights, independent of how many samples you feed.
    print("Batch size is number of training samples, does not create new weights.")


def practice_pooling():
    # 原始图片：4行4列的2D张量，单通道，没有batch维度
    image = torch.tensor([[1.0, 3.0, 2.0, 0.0],
                          [4.0, 6.0, 5.0, 1.0],
                          [0.0, 2.0, 8.0, 4.0],
                          [1.0, 3.0, 6.0, 2.0]])

    # TODO:
    # 1. Print the upper-left 2x2 window, its maximum and its mean.
    # 2. Add N/C axes and apply max_pool2d and avg_pool2d with size=2,stride=2.
    # 3. Print shapes/results and compare with your hand calculation.
    # 4. Explain why pooling has no learned kernel and leaves channels unchanged.
    # 取左上角2*2窗口
    window = image[:2, :2]
    print("upper-left window:", window)
    print("window maximum/mean:", window.max().item(), window.mean().item())
    # 2. 增加batch(N)和通道(C)维度，变成F.max_pool2d要求的 4D格式：(N,C,H,W)
    batch = image[None, None]
    # 最大池化 和 平均池化，窗口2×2，步长stride=2
    max_output = F.max_pool2d(batch, kernel_size=2, stride=2)
    avg_output = F.avg_pool2d(batch, kernel_size=2, stride=2)
    print("pooling input/output shapes:", batch.shape, max_output.shape)
    print("max pooling:", max_output[0, 0])
    print("average pooling:", avg_output[0, 0])
    assert torch.equal(max_output[0, 0], torch.tensor([[6.0, 5.0], [3.0, 8.0]]))
    assert torch.allclose(avg_output[0, 0], torch.tensor([[3.5, 2.0], [1.5, 5.0]]))
    # 两种池化分别在每个通道的空间窗口内按固定规则汇总，
    # 没有需要学习的核权重，也没有把不同通道混合起来。


def multi_channel_corr2d(image, weights, bias):
    """Homework: (Cin,H,W), (Cout,Cin,Kh,Kw), (Cout,) -> (Cout,Hout,Wout)."""
    # TODO:
    # 1. Check dimensions, matching input channels, bias length, dtype/device.
    # 2. For each output channel, sum the 2D correlations over ALL input channels.
    # 3. Add that output channel's bias once, after summing contributions.
    # 4. Stack output maps along dim=0.
    # Use yesterday's manual_corr2d with stride=1,padding=0.
    # Do not call F.conv2d here; it is only used in the supplied reference check.
    if image.ndim != 3 or weights.ndim != 4 or bias.ndim != 1:
        raise ValueError("Expected image (Cin,H,W), weights (Cout,Cin,Kh,Kw), bias (Cout,).")
    cout, cin, kh, kw = weights.shape
    if min(*image.shape, *weights.shape) <= 0:
        raise ValueError("Image and kernel dimensions must be positive.")
    if image.shape[0] != cin or bias.shape[0] != cout:
        raise ValueError("Input channels or output bias length do not match weights.")
    if not image.dtype == weights.dtype == bias.dtype:
        raise TypeError("Image, weights, and bias must use the same dtype.")
    if not image.is_floating_point():
        raise TypeError("Use floating-point tensors.")
    if not image.device == weights.device == bias.device:
        raise ValueError("Image, weights, and bias must use the same device.")
    height, width = image.shape[1:]
    if height < kh or width < kw:
        raise ValueError("The kernel must fit inside the unpadded input.")

    output_maps = []
    for out_channel in range(cout):
        output = image.new_zeros((height - kh + 1, width - kw + 1))
        for in_channel in range(cin):
            contribution = manual_corr2d(
                image[in_channel], weights[out_channel, in_channel],
                stride=1, padding=0,
            )
            output = output + contribution
        # 所有输入通道的贡献加完后，再加当前输出通道的一次偏置。
        output_maps.append(output + bias[out_channel])
    return torch.stack(output_maps, dim=0)


def verify_multi_channel():
    """Provided checks: rectangular kernels and several input/output channels."""
    generator = torch.Generator().manual_seed(7)
    for cin, cout, height, width, kh, kw in [
        (2, 2, 3, 3, 2, 2),
        (3, 4, 5, 6, 2, 3),
        (1, 2, 4, 5, 1, 1),
    ]:
        image = torch.randn((cin, height, width), generator=generator)
        weights = torch.randn((cout, cin, kh, kw), generator=generator)
        bias = torch.randn(cout, generator=generator)
        actual = multi_channel_corr2d(image, weights, bias)
        expected = F.conv2d(image[None], weights, bias)[0]
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
        print("passed Cin/Cout:", cin, cout, "output shape:", actual.shape)


def homework_channels_and_invariance():
    """Homework: extend the working single-channel operation, then interpret demos."""
    # TODO:
    # 1. Implement multi_channel_corr2d.
    # 2. Construct the two-input/two-output example from the knowledge document.
    # 3. Confirm output[0,0,0]=57 and output[1,0,0]=12.
    # 4. Call verify_multi_channel; all three reference checks should pass.
    # 5. Explain equivariance vs invariance and the pooling window-boundary example.
    #
    first = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
    image = torch.stack([first, 10 * first])
    weights = torch.tensor([
        [[[1.0, 0.0], [0.0, -1.0]], [[0.0, 1.0], [1.0, 0.0]]],
        [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
    ])
    bias = torch.tensor([1.0, 0.0])
    output = multi_channel_corr2d(image, weights, bias)
    expected = torch.tensor([[[57.0, 77.0], [117.0, 137.0]],
                             [[12.0, 16.0], [24.0, 28.0]]])
    print("manual multi-channel output:", output)
    assert torch.allclose(output, expected)
    verify_multi_channel()

    # Why sum across input channels:
    # groups=1 的每个输出通道组合全部输入通道，不能把两路贡献当成两个输出。
    # 例如左上输出是 (-4)+60+1=57。
    # Why bias is added once per output:
    # 每个输出通道有一个共享偏置，每个空间位置在通道求和之后加它一次。
    # Translation equivariance means:
    # 输入移动时，输出响应相应移动；stride=1 且无边界干扰时可体现这一性质。
    # Pooling is not completely translation invariant because:
    # 亮点在同一个最大池化窗口内移动可能不改输出，跨过窗口边界就可能改变。
    # 这与所有输入平移后都保持同一个输出的完全不变性不同。
    print("channel aggregation and reference checks passed.")


if __name__ == "__main__":
    print("--- Example 1: channel contributions ---")
    demo_channel_contributions()
    print("\n--- Example 2: translation and pooling ---")
    demo_translation_and_pooling()
    print("\n--- Practice 1: parameter counts ---")
    practice_parameter_counts()
    print("\n--- Practice 2: pooling ---")
    practice_pooling()
    print("\n--- Homework: multi-channel correlation ---")
    homework_channels_and_invariance()
