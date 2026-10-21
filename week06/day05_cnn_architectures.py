"""第6周第5天：经典卷积网络、参数量与多分支结构。"""

import torch
from torch import nn


def count_parameters(model):
    """统计模型注册的参数数量，包含偏置。"""
    return sum(parameter.numel() for parameter in model.parameters())


def build_teaching_lenet():
    """构建教学版网络，不是原始LeNet-5的逐项复现。"""
    return nn.Sequential(
        nn.Conv2d(1, 6, 5),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Flatten(1),
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
    )


class TinyInception(nn.Module):
    """教学用四分支模块：输入8通道，输出16通道。"""

    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(8, 4, 1), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv2d(8, 4, 1), nn.ReLU(),
            nn.Conv2d(4, 6, 3, padding=1), nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(8, 2, 1), nn.ReLU(),
            nn.Conv2d(2, 2, 5, padding=2), nn.ReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(8, 4, 1), nn.ReLU(),
        )

    def forward(self, images):
        # 同一个输入分别进入四条路线，最后沿通道维拼接。
        branches = [self.branch1(images), self.branch2(images),
                    self.branch3(images), self.branch4(images)]
        return torch.cat(branches, dim=1)


def demo_lenet_shapes():
    """示例1：逐层追踪教学版网络的形状。"""
    model = build_teaching_lenet()
    model.eval()
    images = torch.randn(2, 1, 32, 32)
    print("输入形状：", images.shape)
    with torch.no_grad():
        for name, layer in model.named_children():
            images = layer(images)
            print("层编号：", name, "层类型：", type(layer).__name__,
                  "输出形状：", images.shape)
    assert images.shape == (2, 10)
    assert count_parameters(model) == 61706
    print("教学版网络总参数量：", count_parameters(model))


def demo_inception_branches():
    """示例2：观察各分支高宽相同、通道不同。"""
    block = TinyInception()
    block.eval()
    images = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        for index, branch in enumerate(
            [block.branch1, block.branch2, block.branch3, block.branch4], start=1
        ):
            print("分支编号：", index, "输出形状：", branch(images).shape)
        output = block(images)
    assert output.shape == (2, 16, 16, 16)
    assert count_parameters(block) == 450
    print("拼接后的形状：", output.shape, "模块参数量：", count_parameters(block))


def practice_lenet_parameters():
    """练习1：先手算，再逐层核对参数量。"""
    model = build_teaching_lenet()
    # 卷积按输出通道数计算偏置；全连接按输出单元数计算偏置。
    manual_counts = [6 * (1 * 5 * 5 + 1), 16 * (6 * 5 * 5 + 1),
                     400 * 120 + 120, 120 * 84 + 84, 84 * 10 + 10]
    parameter_layers = [(name, layer) for name, layer in model.named_children()
                        if isinstance(layer, (nn.Conv2d, nn.Linear))]
    assert len(parameter_layers) == len(manual_counts) == 5
    for (name, layer), expected in zip(parameter_layers, manual_counts):
        actual = count_parameters(layer)
        assert actual == expected
        print("层编号：", name, "手算参数量：", expected, "实际参数量：", actual)
    assert sum(manual_counts) == count_parameters(model) == 61706
    largest_name, largest_layer = max(parameter_layers,
                                      key=lambda item: count_parameters(item[1]))
    print("参数最多的层编号：", largest_name,
          "参数量：", count_parameters(largest_layer))
    # 卷积在空间位置共享同一组权重，输出数值多不等于参数多。
    # 第一个全连接层需要400×120个独立连接，所以参数占比最大。


def practice_bottleneck_cost():
    """练习2：比较直接大核卷积和先降通道两种方案。"""
    direct = nn.Conv2d(192, 128, 3, padding=1, bias=False)
    bottleneck = nn.Sequential(
        nn.Conv2d(192, 32, 1, bias=False),
        nn.Conv2d(32, 128, 3, padding=1, bias=False),
    )
    images = torch.randn(2, 192, 8, 8)
    with torch.no_grad():
        direct_output = direct(images)
        bottleneck_output = bottleneck(images)
    assert direct_output.shape == bottleneck_output.shape == (2, 128, 8, 8)
    direct_count = 128 * 192 * 3 * 3
    bottleneck_count = 32 * 192 + 128 * 32 * 3 * 3
    assert count_parameters(direct) == direct_count == 221184
    assert count_parameters(bottleneck) == bottleneck_count == 43008
    print("直接卷积输出形状：", direct_output.shape, "权重数：", direct_count)
    print("先降通道输出形状：", bottleneck_output.shape, "权重数：", bottleneck_count)
    print("直接方案与降通道方案的参数量比：", direct_count / bottleneck_count)
    # 相同形状不意味着相同运算：两条路线的随机权重和结构不同。
    # 降通道可能损失信息，效果需要实际训练评估；速度还依赖硬件、访存和算子实现。


def homework_inception_classifier():
    """作业：组合卷积、多分支模块和全局平均池化，完成三分类前向。"""
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),  # 输入通道3，输出通道8，卷积核3*3，padding=1
        nn.ReLU(),
        TinyInception(),
        nn.AdaptiveAvgPool2d((1, 1)), # 对每个通道，把这个通道上所有空间像素求平均值，得到一个数字
        nn.Flatten(1), # 从第一维开始展平，(B,16,1,1) → (B,16)
        nn.Linear(16, 3),
    )
    model.eval()
    images = torch.randn(2, 3, 16, 16)
    targets = torch.tensor([0, 2], dtype=torch.long)
    expected_shapes = [(2, 8, 16, 16), (2, 8, 16, 16), (2, 16, 16, 16),
                       (2, 16, 1, 1), (2, 16), (2, 3)]
    # 这里只核对前向和损失，不进行训练，因此关闭梯度记录。
    with torch.no_grad():
        output = images
        for (name, layer), expected in zip(model.named_children(), expected_shapes):
            output = layer(output)
            assert tuple(output.shape) == expected
            print("分类器层编号：", name, "输出形状：", output.shape)
        # 交叉熵内部处理类别分数，不需要提前使用归一化指数函数。
        loss = nn.CrossEntropyLoss()(output, targets)
        assert loss.ndim == 0 and torch.isfinite(loss).item()
        larger_output = model(torch.randn(2, 3, 20, 20))
        assert larger_output.shape == (2, 3)

    # 模块四个分支分别为36、258、120、36个参数，合计450。
    manual_count = 8 * (3 * 3 * 3 + 1) + (36 + 258 + 120 + 36) + (16 * 3 + 3)
    assert count_parameters(model) == manual_count == 725
    print("三分类交叉熵：", loss.item())
    print("输入改为20×20后的输出形状：", larger_output.shape)
    print("分类器手算与实际参数量：", manual_count, count_parameters(model))
    print("检查通过；随机输入的前向结果不代表模型已学会分类。")

    # 问题1：四条分支为什么可以沿通道拼接？
    # 回答：它们的批量大小和高宽一致，只有通道数不同；拼接后通道数相加。
    # 问题2：1×1卷积为什么不是毫无作用？
    # 回答：它在每个空间位置混合所有输入通道，可以改变通道数，配合激活增加非线性。
    # 问题3：今天的模块与完整GoogLeNet是什么关系？
    # 回答：这里只借用多尺度分支与降通道思想，不是完整网络，也不是论文实验复现。


if __name__ == "__main__":
    torch.manual_seed(7)
    print("--- 示例1：教学版LeNet数据流 ---")
    demo_lenet_shapes()
    print("\n--- 示例2：多分支与通道拼接 ---")
    demo_inception_branches()
    print("\n--- 练习1：逐层参数量 ---")
    practice_lenet_parameters()
    print("\n--- 练习2：降通道的代价 ---")
    practice_bottleneck_cost()
    print("\n--- 作业：多分支分类器 ---")
    homework_inception_classifier()
