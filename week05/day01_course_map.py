"""Week 5 / Day 1: map a learning task to data, model, loss, and evaluation."""

import torch
from torch import nn


def demo_learning_task():
    """Example: a tiny one-feature regression task."""
    X_train = torch.tensor([[-1.0], [0.0], [1.0]]) #训练集输入
    y_train = torch.tensor([[-1.0], [1.0], [3.0]]) # 训练集的标准答案
    X_val = torch.tensor([[0.5]]) # 验证集输入
    y_val = torch.tensor([[2.0]]) # 验证集标准答案

    torch.manual_seed(7)
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = loss_fn(model(X_train), y_train).item()
        val_mae = (model(X_val) - y_val).abs().mean().item()

    print("train samples:", X_train.shape[0])
    print(f"train MSE={train_loss:.4f}, validation MAE={val_mae:.4f}")


def practice_task_components():
    """Practice 1: identify every element of a supervised learning task."""
    X_train = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_train = torch.tensor([[1.0], [2.0], [3.0]])
    model = nn.Linear(2, 1)
    loss_fn = nn.MSELoss()# 只是创建了一个计算器

    # TODO:
    # 1. Predict and then print X_train, y_train, and model(X_train) shapes.
    # 2. Calculate and print the loss before any training.
    # 3. In a comment, label: data, model, loss, and a possible metric.
    # 4. Explain whether this is regression or classification and why.
    print("Practice 1: fill in the task map.")
    # 1.打印shape
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    pred = model(X_train)
    print("model(X_train) shape:", pred.shape)

    # 2.训练前损失
    init_loss = loss_fn(pred, y_train) #开始计算
    print("loss before training:", init_loss.item())

    # 3.组件标注
    # data: X_train(输入特征), y_train(目标标签)
    # model: nn.Linear(2,1) 线性模型
    # loss: MSELoss 均方误差损失
    # metric: MSE or MAE

    # 4.任务类型解释
    # Regression(回归)：目标y是连续数值，不是离散类别


def practice_constant_baseline():
    """Practice 2: evaluate a simple prediction without learning parameters."""
    y_train = torch.tensor([[1.0], [3.0], [5.0]])
    y_val = torch.tensor([[2.0], [4.0]])

    # TODO:
    # 1. Use the mean of y_train as a constant prediction for y_val.
    # 2. Compute validation MAE with PyTorch; print the result.
    # 3. Explain in a comment why a learned model should be compared
    #    against this simple baseline on the same held-out examples.
    # Do not use y_val to choose the constant prediction.
    print("Practice 2: complete the constant baseline.")
    # 1. 取训练集y的均值作为固定预测值
    mean_y = y_train.mean()
    pred_const = torch.full_like(y_val, mean_y)

    # 2. 计算验证集MAE
    val_mae = (pred_const - y_val).abs().mean()
    print("constant baseline val MAE:", val_mae.item())

    # 3.注释解释
    # 基线是最低标准。如果训练后的神经网络，效果还不如直接猜平均值，说明模型完全没用。
    # 必须在同一个验证集上对比，不能偷看验证集数据来设置这个常数。


def homework_analyze_an_application():
    """Homework: train a model, then explain the real-world task boundary."""
    X_train = torch.tensor([
        [-2.0, 0.0], [-1.0, 1.0], [0.0, -1.0],
        [1.0, 0.0], [2.0, 1.0], [1.0, -2.0],
    ])
    y_train = 2 * X_train[:, :1] - X_train[:, 1:] + 1
    X_val = torch.tensor([[-1.5, 0.5], [0.5, -0.5]])
    y_val = 2 * X_val[:, :1] - X_val[:, 1:] + 1

    torch.manual_seed(7)
    model = nn.Linear(2, 1)
    loss_fn = nn.MSELoss()

    # TODO:
    # 1. Create an optimizer and train only on X_train/y_train.
    # 2. After training, report train MSE and validation MAE.
    # 3. Compare validation MAE with a constant y_train-mean baseline.
    # 4. In comments below, pick one real application from Lecture 1 and
    #    describe its input, output, data, model, loss, metric, and one risk.
    # 5. Explain why this tiny synthetic validation set cannot guarantee
    #    good performance on arbitrary real-world data.

    # My application analysis:
    # Input:
    # Output:
    # Data/source:
    # Model:
    # Loss:
    # Metric:
    # Risk/limitation:
    print("Homework: complete the model and application analysis.")
    # 1 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

    # 2 评估
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train)
        train_mse = loss_fn(pred_train, y_train).item()
        pred_val = model(X_val)
        val_mae = (pred_val - y_val).abs().mean().item()
    print("train MSE:", train_mse) # 训练集上的均方误差
    print("val MAE:", val_mae)   # 验证集上的平均绝对误差

    # 3 baseline对比
    mean_y_train = y_train.mean()
    pred_baseline = torch.full_like(y_val, mean_y_train)
    baseline_mae = (pred_baseline - y_val).abs().mean().item()
    print("baseline val MAE:", baseline_mae) # 线模型在验证集上的 MAE


if __name__ == "__main__":
    print("--- Example: a learning task ---")
    demo_learning_task()

    print("\n--- Practice 1: task components ---")
    practice_task_components()

    print("\n--- Practice 2: constant baseline ---")
    practice_constant_baseline()

    print("\n--- Homework: model and application ---")
    homework_analyze_an_application()
