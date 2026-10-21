"""第6周第3天：搭建小型卷积网络，完成横线与竖线图像分类。"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_line_images(per_class=64, seed=7):
    """分别生成带背景噪声的竖线和横线，标签0为竖线、1为横线。"""
    generator = torch.Generator().manual_seed(seed)
    images, targets = [], []
    for label in [0, 1]:
        for _ in range(per_class):
            image = 0.1 * torch.rand((1, 8, 8), generator=generator)
            position = torch.randint(1, 7, (1,), generator=generator).item()
            brightness = 0.8 + 0.2 * torch.rand((), generator=generator).item()
            if label == 0:
                image[0, :, position] = brightness
            else:
                image[0, position, :] = brightness
            images.append(image)
            targets.append(label)
    return torch.stack(images), torch.tensor(targets, dtype=torch.long)


def build_reference_cnn():
    """提供完整示例网络，供观察各层的输入输出形状。"""
    return nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(4, 8, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Flatten(start_dim=1), nn.Linear(32, 2),
    )


def demo_images_and_shapes():
    """示例：预览两类图案，再跟踪随机初始化网络的每层形状。"""
    images, targets = make_line_images(per_class=1, seed=7)
    for image, target in zip(images, targets):
        print("图案类别：", "竖线" if target.item() == 0 else "横线")
        for row in image[0]:
            print("".join("■" if value.item() > 0.5 else "·" for value in row))
    torch.manual_seed(42)
    model = build_reference_cnn()
    current = images
    print("输入形状：", current.shape)
    names = ["第一次卷积", "第一次激活", "第一次池化",
             "第二次卷积", "第二次激活", "第二次池化", "展平", "分类层"]
    for name, layer in zip(names, model):
        current = layer(current)
        print(name, "输出形状：", current.shape)
    print("参考网络参数总数：", sum(p.numel() for p in model.parameters()))


class TinyCNN(nn.Module):
    """练习1：自己定义与示例相同的两组卷积、池化及分类层。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(32, 2)

    def forward(self, X):
        # 第一组：(N,1,8,8) -> (N,4,8,8) -> (N,4,4,4)。
        hidden = self.pool(self.relu(self.conv1(X)))
        # 第二组：(N,4,4,4) -> (N,8,4,4) -> (N,8,2,2)。
        hidden = self.pool(self.relu(self.conv2(hidden)))
        features = self.flatten(hidden)
        # 每个样本32个特征，输出两个原始类别分数。
        return self.classifier(features)


def practice_forward_shapes():
    """练习1：核对自定义模型的输出与参数数目。"""
    images, targets = make_line_images(per_class=2, seed=11)

    torch.manual_seed(42)
    model = TinyCNN()
    logits = model(images)
    print("输入、标签、输出形状：", images.shape, targets.shape, logits.shape)
    assert logits.shape == (4, 2)
    for name, parameter in model.named_parameters():
        print("参数名称：", name, "形状：", parameter.shape)
    parameter_count = sum(p.numel() for p in model.parameters())
    print("参数总数：", parameter_count)
    assert parameter_count == 402
    # 分类层逐个样本处理特征；8个通道各有2×2个数，即每个样本32项。
    # 批次中的样本数是独立的第0维，不合并到单样本特征维。


def practice_one_update():
    """练习2：做一次更新，确认卷积核获得梯度且参数发生变化。"""
    images, targets = make_line_images(per_class=4, seed=13)
    torch.manual_seed(42)

    model = TinyCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    weights_before = model.conv1.weight.detach().clone()
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(images), targets)
    loss_before = loss.item()
    loss.backward()
    for layer in [model.conv1, model.conv2]:
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == layer.weight.shape
        print("卷积核与梯度形状：", layer.weight.shape, layer.weight.grad.shape)
    # 反向只计算梯度，此时参数数值应保持不变。
    assert torch.equal(weights_before, model.conv1.weight)
    optimizer.step()
    changed = not torch.equal(weights_before, model.conv1.weight)
    assert changed
    model.eval()
    with torch.no_grad():
        loss_after = loss_fn(model(images), targets).item()
    print("第一层卷积核是否更新：", changed)
    print(f"更新前损失={loss_before:.6f}，更新后损失={loss_after:.6f}")
    # backward负责求梯度，step负责根据梯度更新已注册参数。


def evaluate_classifier(model, loader):
    """提供评估函数：关闭训练行为与求导图，按样本数累计loss和准确率。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            total_loss += loss_fn(logits, targets).item()
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += len(targets)
    if total_samples == 0:
        raise ValueError("评估数据不能为空。")
    return total_loss / total_samples, total_correct / total_samples


def homework_train_cnn():
    """作业：训练20轮，比较训练/验证指标与常量类别基线。"""
    X_train, y_train = make_line_images(per_class=64, seed=7)
    X_val, y_val = make_line_images(per_class=32, seed=17)
    generator = torch.Generator().manual_seed(202)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16,
                              shuffle=True, generator=generator)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)
    # 使用单独的顺序评估loader，避免记录训练指标时消耗打乱生成器。
    train_eval_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16)
    torch.manual_seed(42)

    model = TinyCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = []
    updates = 0

    def record_metrics(epoch):
        """在同一组当前参数上重新评估两份数据，保留关键日志。"""
        train_loss, train_acc = evaluate_classifier(model, train_eval_loader)
        val_loss, val_acc = evaluate_classifier(model, val_loader)
        history.append(dict(epoch=epoch, updates=updates, train_loss=train_loss,
                            train_acc=train_acc, val_loss=val_loss, val_acc=val_acc))
        print(
            f"轮次={epoch:2d}，累计更新={updates:3d} | "
            f"训练损失={train_loss:.6f}，准确率={train_acc:.3f} | "
            f"验证损失={val_loss:.6f}，准确率={val_acc:.3f}"
        )

    record_metrics(0)
    for epoch in range(1, 21):
        # 记录指标会切到评估模式，每轮开始时重新切回训练模式。
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            updates += 1
        if epoch in [1, 5, 20]:
            record_metrics(epoch)
    assert updates == 160

    # 各选4张验证图，方便同时观察两类；这里只作展示，不据此调参。
    preview_indices = torch.cat([torch.where(y_val == label)[0][:4] for label in [0, 1]])
    model.eval()
    with torch.no_grad():
        logits = model(X_val[preview_indices])
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
    names = ["竖线", "横线"]
    for index, prediction, probability in zip(preview_indices, predictions, probabilities):
        print(
            f"样本{index.item()}：真实={names[y_val[index].item()]}，"
            f"预测={names[prediction.item()]}，"
            f"竖线概率={probability[0].item():.4f}，横线概率={probability[1].item():.4f}"
        )

    # argmax在最大计数平票时选第一个类别，因此本次固定选择类别0。
    majority_class = torch.bincount(y_train, minlength=2).argmax().item()
    baseline_accuracy = (y_val == majority_class).float().mean().item()
    print("多数类别基线选择：", names[majority_class], "验证准确率：", baseline_accuracy)
    # 展平后的32项来自8张2×2特征图，样本轴单独保留。
    # 验证数据不参与反向和更新，用于观察未参与训练的图案表现。
    # 合成数据只有两种简单图形及有限噪声，未覆盖自然图像、断线、旋转等变化；
    # 较高验证准确率不能直接说明模型具备真实图像识别能力。
    return history


if __name__ == "__main__":
    print("--- 示例：图案与每层形状 ---")
    demo_images_and_shapes()
    print("\n--- 练习1：模型与参数 ---")
    practice_forward_shapes()
    print("\n--- 练习2：一次参数更新 ---")
    practice_one_update()
    print("\n--- 作业：完整分类训练 ---")
    homework_train_cnn()
