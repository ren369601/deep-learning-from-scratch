"""第6周第4天：卷积梯度、空洞采样与转置卷积贡献块。"""

import torch
from torch import nn
from torch.nn import functional as F


def demo_shared_kernel_gradient():
    """示例1：共享核在四个输出位置的梯度贡献会相加。"""
    # 1. 构造图片 image，shape = (N=1, C_in=1, H=3, W=3)
    image = torch.arange(1, 10, dtype=torch.float32).reshape(1, 1, 3, 3)
    # 数值：
    # [[1,2,3],
    #  [4,5,6],
    #  [7,8,9]]

    # 2. 卷积核 kernel，shape=(C_out=1, C_in=1, kh=2, kw=2)
    # requires_grad=True：要计算这个核的梯度（卷积权重的梯度）
    kernel = torch.tensor([[[[1.0, 0.0],
                             [0.0, -1.0]]]], requires_grad=True)

    # 3. 普通2D互相关卷积 F.conv2d
    # stride默认=1，padding=0，核大小2×2
    output = F.conv2d(image, kernel)
    # 输入3×3，核2×2，输出尺寸：(3-2)+1 = 2 → output是 1,1,2,2
    # 输出一共4个数字（2行2列），对应核滑动到4个窗口算出的值

    # 4. output.sum().backward() 【核心！】
    # output.sum()：把输出张量全部4个数字加起来，得到一个标量。
    # backward()：反向求导。
    # 因为我们对output全部元素求和，等价于：每个输出位置的上游梯度 = 1
    # 卷积**共享权重**：同一个kernel，在4次滑动窗口里反复被使用。
    # ➡️ 核的梯度 = 4个窗口对应的image小块，全部加在一起！
    # 这就是标题：共享核，多个输出位置的梯度贡献相加
    output.sum().backward()

    # 5. 预先算好的理论梯度结果，用来验证
    expected = torch.tensor([[[[12.0, 16.0],
                                [24.0, 28.0]]]])

    # 打印卷积输出结果
    print("卷积输出：", output[0, 0])
    # 打印卷积核的梯度 kernel.grad，取第0输出通道、第0输入通道的2×2矩阵
    print("共享卷积核梯度：", kernel.grad[0, 0])

    # 断言：判断算出的梯度和理论值是否几乎相等，不对就报错
    assert torch.allclose(kernel.grad, expected)


def demo_dilated_sampling():
    """示例2：同一个2×2全1核，取连续位置与隔格位置。"""
    # image shape: [N=1, Cin=1, H=5, W=5]
    # 数值： 1, 2, 3, 4, 5
    #        6, 7, 8, 9,10
    #       11,12,13,14,15
    #       16,17,18,19,20
    #       21,22,23,24,25
    image = torch.arange(1, 26, dtype=torch.float32).reshape(1, 1, 5, 5)
    # 2×2卷积核，全部是1
    kernel = torch.ones(1, 1, 2, 2)

    # 普通卷积 dilation=1（默认），核是紧贴的2×2方块
    ordinary = F.conv2d(image, kernel)
    # 空洞卷积 dilation=2：核里面元素之间**隔1个像素**，感受野变大，核的参数数量不变！
    dilated = F.conv2d(image, kernel, dilation=2)

    print("普通卷积左上值与形状：", ordinary[0, 0, 0, 0].item(), ordinary.shape)
    print("空洞卷积左上值与形状：", dilated[0, 0, 0, 0].item(), dilated.shape)

    # 校验：普通卷积左上角窗口是 [[1,2],[6,7]]，求和 =1+2+6+7 =16
    assert ordinary[0, 0, 0, 0].item() == 16
    # dilation=2，2×2核采样位置：(0,0)、(0,2)、(2,0)、(2,2)
    # 四个像素：1,3,11,13 求和=1+3+11+13 = 28
    assert dilated[0, 0, 0, 0].item() == 28


def demo_transposed_overlap():
    """示例3：贡献块重叠时相加，并观察stride对输出大小的影响。"""
    # 输入image：N=1,Cin=1,H=2,W=2
    # [[1,2],
    #  [3,4]]
    image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    # 2×2卷积核
    kernel = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    # stride默认=1 的转置卷积 ConvTranspose2d
    output = F.conv_transpose2d(image, kernel)
    # 预期输出 3×3矩阵，就是我们前面讲的：每个元素 × kernel，铺画布，重叠区域相加
    expected = torch.tensor([[1.0, 4.0, 4.0],
                             [6.0, 20.0, 16.0],
                             [9.0, 24.0, 16.0]])

    print("步长1的转置卷积：", output[0, 0])
    # stride=2 的转置卷积，直接打印输出的shape（4×4，块之间无重叠）
    print("步长2的输出形状：", F.conv_transpose2d(image, kernel, stride=2).shape)
    # 判断结果完全相等
    assert torch.equal(output[0, 0], expected)



def practice_kernel_update():
    """练习1：让1×1卷积核做一次可手算的梯度更新。"""
    image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    weight = torch.tensor([[[[0.5]]]], requires_grad=True)
    targets = torch.zeros_like(image)

    optimizer = torch.optim.SGD([weight], lr=0.01)
    original_weight = weight.detach().clone()
    optimizer.zero_grad()
    output = F.conv2d(image, weight)
    loss = F.mse_loss(output, targets)
    loss.backward()

    # 目标为0时，损失为w²乘输入平方的均值，对w求导得到下式。
    expected_gradient = 2 * original_weight * image.square().mean()
    print("更新前损失：", loss.item())
    print("自动求导与手算梯度：", weight.grad.item(), expected_gradient.item())
    assert torch.allclose(weight.grad, expected_gradient)
    assert torch.equal(weight.detach(), original_weight)

    # 反向传播只计算梯度；优化器执行更新后，权重才真正改变。
    optimizer.step()
    with torch.no_grad():
        new_loss = F.mse_loss(F.conv2d(image, weight), targets)
    assert torch.allclose(weight.detach(), original_weight - 0.01 * expected_gradient)
    assert new_loss.item() < loss.item()
    print("更新后权重：", weight.item())
    print("更新后损失：", new_loss.item())


def practice_dilation_shapes():
    """练习2：对齐权重后比较普通卷积和空洞卷积。"""
    image = torch.arange(1, 50, dtype=torch.float32).reshape(1, 1, 7, 7)
    ordinary = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    dilated = nn.Conv2d(1, 1, 3, padding=2, dilation=2, bias=False)

    with torch.no_grad():
        ordinary.weight.fill_(1.0)
        dilated.weight.fill_(1.0)
        ordinary_output = ordinary(image)
        dilated_output = dilated(image)
        ordinary_squared = ordinary(image.square())
        dilated_squared = dilated(image.square())

    for name, layer, output in [("普通卷积", ordinary, ordinary_output),
                                ("空洞卷积", dilated, dilated_output)]:
        span = layer.dilation[0] * (layer.kernel_size[0] - 1) + 1
        print(name, "输出形状：", output.shape,
              "核参数数：", layer.weight.numel(), "有效覆盖跨度：", span)
    assert ordinary_output.shape == dilated_output.shape == image.shape
    assert ordinary.weight.numel() == dilated.weight.numel() == 9
    print("等差输入的中心结果：", ordinary_output[0, 0, 3, 3].item(),
          dilated_output[0, 0, 3, 3].item())
    print("平方输入的中心结果：", ordinary_squared[0, 0, 3, 3].item(),
          dilated_squared[0, 0, 3, 3].item())
    assert ordinary_output[0, 0, 3, 3].item() == 225
    assert dilated_output[0, 0, 3, 3].item() == 225
    assert ordinary_squared[0, 0, 3, 3].item() == 5925
    assert dilated_squared[0, 0, 3, 3].item() == 6825
    # 等差输入的对称位置求和碰巧相等；平方输入暴露了采样位置的区别。
    # 空洞卷积并未增加权重数量，只扩大了这些权重在输入上的覆盖跨度。


def manual_transpose2d(image, kernel, stride=1):
    """作业：单通道、无填充、无偏置，把输入乘核贡献块叠加到输出。"""
    for name, tensor in [("输入", image), ("卷积核", kernel)]:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}必须是张量。")
        if tensor.ndim != 2 or tensor.numel() == 0:
            raise ValueError(f"{name}必须是非空二维张量。")
        if not tensor.is_floating_point():
            raise TypeError(f"{name}必须是浮点张量。")
    if image.dtype != kernel.dtype or image.device != kernel.device:
        raise ValueError("输入和卷积核的数据类型、设备必须一致。")
    if type(stride) is not int or stride <= 0:
        raise ValueError("步长必须是正整数，不能是布尔值。")

    height, width = image.shape
    kh, kw = kernel.shape
    output = image.new_zeros(((height - 1) * stride + kh,
                              (width - 1) * stride + kw))
    # 每个输入数值生成一整块贡献；落在相同输出位置的贡献必须累加。
    for row in range(height):
        for column in range(width):
            start_row = row * stride
            start_column = column * stride
            output[start_row:start_row + kh, start_column:start_column + kw] += (
                image[row, column] * kernel
            )
    return output


def verify_manual_transpose():
    """提供数值核对：比较重叠、不重叠与长方形情况。"""
    generator = torch.Generator().manual_seed(7)
    cases = [(2, 2, 2, 2, 1), (2, 2, 2, 2, 2), (2, 3, 3, 2, 2)]
    for height, width, kh, kw, stride in cases:
        image = torch.randn((height, width), generator=generator)
        kernel = torch.randn((kh, kw), generator=generator)
        actual = manual_transpose2d(image, kernel, stride)
        reference = F.conv_transpose2d(image[None, None], kernel[None, None],
                                       stride=stride)[0, 0]
        assert actual.shape == reference.shape
        assert torch.allclose(actual, reference, atol=1e-5, rtol=1e-5)
        print("核对通过，步长：", stride, "输出形状：", actual.shape)


def homework_transposed_convolution():
    """作业：手写转置运算，核对贡献块，并解释尺寸选择。"""
    image = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    kernel = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = torch.tensor([[1.0, 4.0, 4.0],
                             [6.0, 20.0, 16.0],
                             [9.0, 24.0, 16.0]])
    output = manual_transpose2d(image, kernel)
    assert torch.equal(output, expected)
    print("手写步长1输出：", output)

    # 步长2时，输出边长为(2-1)×2+2=4。
    spaced_output = manual_transpose2d(image, kernel, stride=2)
    assert spaced_output.shape == (4, 4)
    reference = F.conv_transpose2d(image[None, None], kernel[None, None], stride=2)
    assert torch.equal(spaced_output, reference[0, 0])
    print("预测步长2输出为4×4，实际输出：", spaced_output)
    verify_manual_transpose()

    image = torch.ones(1, 1, 3, 3)
    kernel = torch.ones(1, 1, 3, 3)
    for output_padding in (0, 1):
        output = F.conv_transpose2d(image, kernel, stride=2, padding=1,
                                    output_padding=output_padding)
        expected_size = 5 + output_padding
        assert output.shape == (1, 1, expected_size, expected_size)
        print("输出尺寸补偿：", output_padding, "输出形状：", output.shape)

    # 重叠时必须相加：同一个输出位置可能收到多个输入像素的贡献。
    # 转置卷积是卷积线性算子的转置，不是逆；普通卷积可能已经丢失信息。
    # 输出尺寸补偿用于消除步长带来的尺寸歧义，不是直接在结果旁补一圈0。
    print("总结：贡献重叠要相加；转置不是求逆；输出尺寸补偿用于选择输出大小。")


if __name__ == "__main__":
    print("--- 示例1：卷积核梯度 ---")
    demo_shared_kernel_gradient()
    print("\n--- 示例2：空洞采样 ---")
    demo_dilated_sampling()
    print("\n--- 示例3：转置贡献块 ---")
    demo_transposed_overlap()
    print("\n--- 练习1：一次梯度更新 ---")
    practice_kernel_update()
    print("\n--- 练习2：空洞卷积比较 ---")
    practice_dilation_shapes()
    print("\n--- 作业：手写转置卷积 ---")
    homework_transposed_convolution()
