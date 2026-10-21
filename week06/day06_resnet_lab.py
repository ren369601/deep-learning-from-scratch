"""第6周第6天：残差块、形状对齐及普通网络与残差网络的对照。"""

from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from day03_cnn_classifier import make_line_images, evaluate_classifier


class ResidualBlock(nn.Module):
    """两层卷积的教学残差块；形状变化时使用投影捷径。"""

    def __init__(self, in_channels, out_channels, stride=1, use_skip=True):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity() if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.use_skip = use_skip
        self.relu = nn.ReLU()

    def forward(self, images):
        features = self.main(images)
        if self.use_skip:
            shortcut = self.shortcut(images)
            # 显式核对形状，避免广播使错误的加法悄悄运行。
            assert features.shape == shortcut.shape
            features = features + shortcut
        return self.relu(features)


class TinyClassifier(nn.Module):
    """两组网络采用相同层结构，唯一切换项为是否加上恒等捷径。"""

    def __init__(self, use_skip):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        # 对照中固定通道与空间大小，所以所有捷径均无参数。
        self.blocks = nn.Sequential(
            *(ResidualBlock(8, 8, use_skip=use_skip) for _ in range(3))
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
                                  nn.Linear(8, 2))

    def forward(self, images):
        return self.head(self.blocks(self.stem(images)))


def demo_residual_numbers():
    """示例1：把原值与修正量相加，注意最后的激活。"""
    images = torch.tensor([-2.0, 3.0])
    correction = torch.tensor([0.5, -1.0])
    print("输入：", images, "修正量：", correction)
    print("相加结果：", images + correction)
    print("最终激活后：", torch.relu(images + correction))
    assert torch.equal(torch.relu(images + correction), torch.tensor([0.0, 2.0]))


def demo_gradient_paths():
    """示例2：用标量展示恒等捷径在局部导数中增加的一项。"""
    plain_input = torch.tensor(2.0, requires_grad=True)
    skip_input = torch.tensor(2.0, requires_grad=True)
    # 此处输入为正，激活导数为1，便于单独观察两条路径。
    torch.relu(0.1 * plain_input).backward()
    torch.relu(skip_input + 0.1 * skip_input).backward()
    assert torch.allclose(plain_input.grad, torch.tensor(0.1))
    assert torch.allclose(skip_input.grad, torch.tensor(1.1))
    print("普通路径输入梯度：", plain_input.grad.item())
    print("残差路径输入梯度：", skip_input.grad.item())
    # 这是局部例子，不代表所有网络的梯度都会变大或永不消失。


def practice_block_shapes():
    """练习1已完成：检查恒等捷径与投影捷径的形状和梯度。"""
    for in_channels, out_channels, stride, size in [(8, 8, 1, 8), (8, 16, 2, 9)]:
        block = ResidualBlock(in_channels, out_channels, stride)
        images = torch.randn(2, in_channels, size, size, requires_grad=True)
        output = block(images)
        expected_size = (size + stride - 1) // stride
        assert output.shape == (2, out_channels, expected_size, expected_size)
        output.square().mean().backward()
        assert images.grad is not None and torch.isfinite(images.grad).all()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in block.parameters())
        print("输入形状：", images.shape, "输出形状：", output.shape,
              "捷径类型：", type(block.shortcut).__name__)
    # 9×9经步长2和填充1的3×3卷积得到5×5，不是直接向下取整成4。


def practice_zero_correction():
    """练习2已完成：主分支为零时，验证输出是输入经过激活。"""
    block = ResidualBlock(2, 2)
    block.eval()
    with torch.no_grad():
        # 归一化层保持初始状态，两个卷积核清零，确保修正量为零。
        block.main[0].weight.zero_()
        block.main[3].weight.zero_()
        images = torch.tensor([[[[-2.0, 3.0]], [[1.0, -4.0]]]])
        assert torch.equal(block.main(images), torch.zeros_like(images))
        output = block(images)
        assert torch.equal(output, torch.relu(images))
    print("零修正量时的输出：", output)
    # 相加前等于输入，但激活会截断负数，不能无条件说整个块恒等。


def train_one_model(model, train_images, train_targets, val_loader, seed, epochs=12):
    """固定批次打乱种子，记录更新前后及中间轮次的训练和验证指标。"""
    train_loader = DataLoader(TensorDataset(train_images, train_targets), batch_size=16,
                              shuffle=True, generator=torch.Generator().manual_seed(seed))
    train_eval = DataLoader(TensorDataset(train_images, train_targets), batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    history = []

    def record(epoch):
        train_loss, train_acc = evaluate_classifier(model, train_eval)
        val_loss, val_acc = evaluate_classifier(model, val_loader)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc))
        print(f"轮次={epoch:2d} | 训练损失={train_loss:.6f}，准确率={train_acc:.3f} | "
              f"验证损失={val_loss:.6f}，准确率={val_acc:.3f}")

    record(0)
    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), targets)
            assert torch.isfinite(loss).item()
            loss.backward()
            optimizer.step()
        if epoch in (1, 4, epochs):
            record(epoch)
    return history


def homework_compare_networks():
    """作业已完成：三个初始化种子下做只切换残差相加的配对对照。"""
    train_images, train_targets = make_line_images(per_class=64, seed=7)
    val_images, val_targets = make_line_images(per_class=32, seed=17)
    val_loader = DataLoader(TensorDataset(val_images, val_targets), batch_size=32)
    results = []
    for seed in (7, 17, 27):
        torch.manual_seed(seed)
        plain = TinyClassifier(use_skip=False)
        residual = deepcopy(plain)
        for block in residual.blocks:
            block.use_skip = True
        # 复制后所有参数和归一化缓冲区相同，唯一区别是前向中的加法。
        assert all(torch.equal(value, residual.state_dict()[name])
                   for name, value in plain.state_dict().items())
        for name, model in [("普通网络", plain), ("残差网络", residual)]:
            count = sum(parameter.numel() for parameter in model.parameters())
            assert count == 3658
            print("\n初始化种子：", seed, "模型：", name, "参数量：", count)
            history = train_one_model(model, train_images, train_targets, val_loader,
                                      seed=202, epochs=12)
            results.append(dict(seed=seed, model=name, **history[-1]))

    print("\n--- 最终配对结果 ---")
    for result in results:
        print(f"种子={result['seed']}，{result['model']}："
              f"验证损失={result['val_loss']:.6f}，验证准确率={result['val_acc']:.3f}")
    for name in ("普通网络", "残差网络"):
        scores = torch.tensor([row["val_acc"] for row in results if row["model"] == name])
        print(name, "三个种子的验证准确率均值：", scores.mean().item(),
              "总体标准差：", scores.std(unbiased=False).item())
    majority = torch.bincount(train_targets).argmax()
    print("多数类别基线验证准确率：", (val_targets == majority).float().mean().item())
    # 结论：按实际指标比较，不强行要求残差网络胜出。
    # 三个种子与简单合成任务只能演示实验流程，不能复现论文或证明普遍优势。
    # 本次验证集用于教学观察，未另设独立测试集，不将其称为最终测试成绩。
    return results


if __name__ == "__main__":
    # 小张量使用单线程减少线程调度开销，便于普通电脑复现。
    torch.set_num_threads(1)
    torch.manual_seed(7)
    print("--- 示例1：输入加修正量 ---")
    demo_residual_numbers()
    print("\n--- 示例2：两条梯度通路 ---")
    demo_gradient_paths()
    print("\n--- 练习1：恒等与投影捷径 ---")
    practice_block_shapes()
    print("\n--- 练习2：零修正量检查 ---")
    practice_zero_correction()
    print("\n--- 作业：普通网络与残差网络 ---")
    homework_compare_networks()
