"""Week 4 / Day 6: train and validate the Day 5 PyTorch MLP."""

import torch
from torch import nn

from day05_pytorch_mlp import TwoLayerMLP


def make_xor_data(seed=7, train_per_corner=20, val_per_corner=5):
    """Create separate noisy samples around each of the four XOR corners."""
    centers = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    train_points, val_points, train_labels, val_labels = [], [], [], []

    for center, label in zip(centers, labels):
        count = train_per_corner + val_per_corner
        points = center + 0.08 * torch.randn((count, 2), generator=generator)
        train_points.append(points[:train_per_corner])
        val_points.append(points[train_per_corner:])
        train_labels.append(label.repeat(train_per_corner))
        val_labels.append(label.repeat(val_per_corner))

    return (
        torch.cat(train_points),
        torch.cat(train_labels),
        torch.cat(val_points),
        torch.cat(val_labels),
    )


def evaluate(model, X, targets, loss_fn):
    """Return loss and accuracy without building a gradient graph."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = loss_fn(logits, targets).item()
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return loss, accuracy


def demo_data_and_evaluation():
    """Show the split and evaluate a model before training."""
    X_train, y_train, X_val, y_val = make_xor_data()
    torch.manual_seed(42)
    model = TwoLayerMLP(2, 16, 2)
    loss_fn = nn.CrossEntropyLoss()
    train_loss, train_acc = evaluate(model, X_train, y_train, loss_fn)
    val_loss, val_acc = evaluate(model, X_val, y_val, loss_fn)
    print("train shapes:", X_train.shape, y_train.shape)
    print("val shapes:  ", X_val.shape, y_val.shape)
    print(f"before training | train loss={train_loss:.4f}, acc={train_acc:.2f}")
    print(f"before training | val   loss={val_loss:.4f}, acc={val_acc:.2f}")


def practice_one_train_step():
    """练习 1：做一次参数更新并重新评估。"""
    X_train, y_train, X_val, y_val = make_xor_data()
    torch.manual_seed(42)
    model = TwoLayerMLP(2, 16, 2)
    loss_fn = nn.CrossEntropyLoss()

    # TODO:
    # 1. Create Adam(model.parameters(), lr=0.03).
    # 2. Record train and validation loss before the update with evaluate().
    # 3. model.train() -> zero_grad -> forward -> loss -> backward -> step.
    # 4. Re-evaluate train and validation loss after the update.
    # 5. Print both sets of before/after metrics. Do not assume one step
    #    must improve every validation metric.

    # 1. 定义Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    # 2. 更新前评估
    train_loss_before, train_acc_before = evaluate(model, X_train, y_train, loss_fn)
    val_loss_before, val_acc_before = evaluate(model, X_val, y_val, loss_fn)

    # 3. 一轮训练更新
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()

    # 4. 更新后重新评估
    train_loss_after, train_acc_after = evaluate(model, X_train, y_train, loss_fn)
    val_loss_after, val_acc_after = evaluate(model, X_val, y_val, loss_fn)

    # 5. 打印
    print("===== 更新前 =====")
    print(f"train loss={train_loss_before:.4f}, acc={train_acc_before:.2f}")
    print(f"val   loss={val_loss_before:.4f}, acc={val_acc_before:.2f}")
    print("===== 更新后 =====")
    print(f"train loss={train_loss_after:.4f}, acc={train_acc_after:.2f}")
    print(f"val   loss={val_loss_after:.4f}, acc={val_acc_after:.2f}")



def practice_validation_does_not_update():
    """练习 2：验证 evaluate() 不改变参数。"""
    _, _, X_val, y_val = make_xor_data()
    torch.manual_seed(42)
    model = TwoLayerMLP(2, 16, 2)
    loss_fn = nn.CrossEntropyLoss()

    # TODO:
    # 1. Clone all model parameters before evaluation.
    # 2. Call evaluate(model, X_val, y_val, loss_fn).
    # 3. Compare each saved parameter with the current parameter using
    #    torch.equal; print whether all stayed unchanged.
    # 4. Explain the different jobs of model.eval() and torch.no_grad().
    # 1. 克隆保存所有参数
    old_params = [p.clone() for p in model.parameters()]
    # 2. 执行评估
    evaluate(model, X_val, y_val, loss_fn)
    # 3. 对比
    all_same = True
    for old_p, new_p in zip(old_params, model.parameters()):
        if not torch.equal(old_p, new_p):
            all_same = False
    print("所有参数是否不变：", all_same)
    # 4. 解释
    # model.eval()：改变Dropout/BatchNorm层行为，不关闭梯度计算
    # torch.no_grad()：关闭自动求导，不构建计算图，不计算梯度，不会更新参数


def homework_train_and_validate():
    """作业：记录完整训练和验证过程。"""
    X_train, y_train, X_val, y_val = make_xor_data()
    torch.manual_seed(42)
    model = TwoLayerMLP(2, 16, 2)
    loss_fn = nn.CrossEntropyLoss()

    # TODO:
    # 1. Create Adam(model.parameters(), lr=0.03).
    # 2. Train for 200 epochs using only X_train/y_train for updates.
    # 3. After each step, use evaluate() to record both train and val
    #    loss/accuracy with the newly updated model.
    # 4. Print metrics at epochs 1, 50, 100, 150, and 200.
    # 5. Print final predictions on the four clean XOR corners.
    # 6. In a comment, explain why validation results do not prove
    #    performance on arbitrary data outside these four neighborhoods.
    # 1. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    epochs = 200
    for epoch in range(1, epochs + 1):
        # 训练一步
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        # 每轮训练完评估
        train_loss, train_acc = evaluate(model, X_train, y_train, loss_fn)
        val_loss, val_acc = evaluate(model, X_val, y_val, loss_fn)
        # 指定epoch打印
        if epoch in [1, 50, 100, 150, 200]:
            print(
                f"Epoch {epoch:3d} | train loss:{train_loss:.4f}, acc:{train_acc:.2f} | val loss:{val_loss:.4f}, acc:{val_acc:.2f}")
    # 5. 四个干净XOR角点预测
    corners = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    model.eval()
    with torch.no_grad():
        pred = model(corners).argmax(dim=1)
    print("四个角预测结果：", pred)
    # 6. 注释
    # 我们的训练、验证样本都只在四个XOR角附近，属于局部分布。
    # 验证集只是从这个局部分布里抽出来的样本，只能评估在这个局部分布的效果。
    # 如果样本远离这四个角，不属于当前分布，模型没有见过这类数据，预测不可靠，所以不能代表任意数据性能。


if __name__ == "__main__":
    print("--- 数据与初始评估示例 ---")
    demo_data_and_evaluation()

    print("\n--- 练习 1：一次训练更新 ---")
    practice_one_train_step()

    print("\n--- 练习 2：验证不更新参数 ---")
    practice_validation_does_not_update()

    print("\n--- 作业：训练与验证 ---")
    homework_train_and_validate()
