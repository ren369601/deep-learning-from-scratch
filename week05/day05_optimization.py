"""Week 5 / Day 5: learning rates, mini-batches, and controlled experiments."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def demo_learning_rates():
    """Example: the same quadratic and starting point, three step sizes."""
    for learning_rate in [0.1, 1.0, 1.2]:
        weight = 1.0
        losses = [(weight - 3) ** 2]
        for _ in range(5):
            gradient = 2 * (weight - 3)
            weight -= learning_rate * gradient
            losses.append((weight - 3) ** 2)
        print("lr:", learning_rate, "losses:", [round(v, 4) for v in losses])


def demo_batch_sizes():
    """Example: ten samples make three batches when batch_size is four."""
    X = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    y = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)
    for batch_id, (X_batch, y_batch) in enumerate(loader):
        print("batch:", batch_id, "shapes:", X_batch.shape, y_batch.shape)
    print("updates per epoch if stepping once per batch:", len(loader))


def make_data(seed=7):
    """Create separate balanced splits with deliberately different feature scales."""
    generator = torch.Generator().manual_seed(seed)

    def split(per_class):
        negative = -1 + 0.35 * torch.randn((per_class, 2), generator=generator)
        positive = 1 + 0.35 * torch.randn((per_class, 2), generator=generator)
        X = torch.cat([negative, positive])
        X = X * torch.tensor([1.0, 100.0]) + torch.tensor([10.0, -50.0])
        y = torch.cat([torch.zeros((per_class, 1)), torch.ones((per_class, 1))])
        return X, y

    X_train, y_train = split(32)
    X_val, y_val = split(16)
    return X_train, y_train, X_val, y_val


def practice_standardization():
    """Practice 1: estimate feature statistics only from training data."""
    X_train, y_train, X_val, y_val = make_data()

    # TODO:
    # 1. Print the original per-feature mean and std of X_train.
    # 2. Compute mean/std along dim=0 with keepdim=True and correction=0.
    # 3. Clamp std to at least 1e-6, then transform both splits using it.
    # 4. Print both shapes and the transformed training mean/std.
    # 5. Explain why X_val must not provide its own fitted statistics.
    # Do not standardize the 0/1 classification labels.
    mean = X_train.mean(dim=0, keepdim=True)
    raw_std = X_train.std(dim=0, correction=0, keepdim=True)
    print("original mean:", mean)
    print("original std:", raw_std)

    std = raw_std.clamp_min(1e-6)
    train_scaled = (X_train - mean) / std
    val_scaled = (X_val - mean) / std
    print("scaled train/val shapes:", train_scaled.shape, val_scaled.shape)
    print("scaled training mean:", train_scaled.mean(dim=0))
    print("scaled training std:", train_scaled.std(dim=0, correction=0))
    # 验证输入沿用训练均值和标准差，模拟处理未来未见数据。
    # 如果重新拟合验证统计量，会利用验证信息，也会改变模型输入的坐标系。
    # 标签表示类别，保持原来的 0/1。


def practice_one_minibatch_epoch():
    """Practice 2: train one epoch and count updates rather than just epochs."""
    X_train, y_train, _, _ = make_data()
    mean = X_train.mean(0, keepdim=True)
    std = X_train.std(0, correction=0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    generator = torch.Generator().manual_seed(202)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16,
                        shuffle=True, generator=generator)
    torch.manual_seed(42)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    loss_fn = nn.BCEWithLogitsLoss()

    # TODO:
    # 1. Set train mode and loop over loader.
    # 2. Clear gradients, forward, compute BCE, backward, step in each batch.
    # 3. Count parameter updates and samples processed; print both.
    # 4. Evaluate full-training BCE after all updates using eval/no_grad.
    # 5. Explain why one epoch has four updates in this example.
    model.train()
    updates = 0
    samples_processed = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        updates += 1
        samples_processed += X_batch.shape[0]

    # 所有 batch 更新结束后，重新评估当前模型在完整训练集上的表现。
    train_loss, train_accuracy = evaluate(model, X_train, y_train, loss_fn)
    print("updates:", updates, "samples processed:", samples_processed)
    print(f"after one epoch | train BCE={train_loss:.6f}, acc={train_accuracy:.3f}")
    # 64 个样本、每批 16 个，一轮有 4 批，因此更新 4 次。


def evaluate(model, X, y, loss_fn):
    """Provided evaluation: all metrics correspond to the current parameters."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = loss_fn(logits, y).item()
        accuracy = ((logits >= 0).float() == y).float().mean().item()
    return loss, accuracy


def run_experiment(kind, learning_rate, epochs=50):
    """Provided helper: recreate data, initialization, optimizer, and batch order."""
    X_train, y_train, X_val, y_val = make_data(seed=7)
    mean = X_train.mean(0, keepdim=True)
    std = X_train.std(0, correction=0, keepdim=True).clamp_min(1e-6)
    X_train, X_val = (X_train - mean) / std, (X_val - mean) / std

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    if kind == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif kind == "momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif kind == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("kind must be sgd, momentum, or adam")
    generator = torch.Generator().manual_seed(202)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16,
                        shuffle=True, generator=generator)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    updates = 0

    for epoch in range(epochs + 1):
        if epoch > 0:
            model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(X_batch), y_batch)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite loss: inspect data and learning rate.")
                loss.backward()
                optimizer.step()
                updates += 1
        train_loss, train_acc = evaluate(model, X_train, y_train, loss_fn)
        val_loss, val_acc = evaluate(model, X_val, y_val, loss_fn)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, updates=updates))
    return history


def homework_optimizer_comparison():
    """Homework: compare optimizers and learning rates with fixed conditions."""
    configurations = [
        ("sgd", 0.03),
        ("adam", 0.03),
        ("sgd", 0.003),
        ("sgd", 0.3),
        ("momentum", 0.03),
    ]

    # TODO:
    # 1. Call run_experiment for each configuration; keep each returned history.
    # 2. Print train/val BCE and accuracy plus updates at epochs 0, 10, 50.
    # 3. Compare SGD vs Adam at lr=0.03, then the three SGD learning rates.
    # 4. Describe the observed differences without assuming a universal winner.
    # 5. Optional: add ("momentum", 0.03) as another separate experiment.
    #
    histories = {}
    for kind, learning_rate in configurations:
        history = run_experiment(kind, learning_rate)
        histories[(kind, learning_rate)] = history
        print(f"\noptimizer={kind}, lr={learning_rate}")
        for epoch in [0, 10, 50]:
            row = history[epoch]
            print(
                f"epoch={row['epoch']:2d}, updates={row['updates']:3d} | "
                f"train BCE={row['train_loss']:.6f}, acc={row['train_acc']:.3f} | "
                f"val BCE={row['val_loss']:.6f}, acc={row['val_acc']:.3f}"
            )

    # SGD/Adam observation:
    # 本次固定 lr=0.03，Adam 在 50 轮后得到更低的训练和验证 BCE。
    # 这只描述本数据、初始化和预算，不能推论 Adam 对所有任务都最好。
    # Learning-rate observation:
    # 本次 SGD 的 0.003 下降较慢，0.3 比 0.03 得到更低的最终 BCE。
    # 这里 0.3 没有发散，但不能据此认为学习率越大越好；
    # 前面的二次函数示例已经展示了过大步长造成的发散。
    # Why every run restarts:
    # 每次重建模型、优化器和打乱生成器，固定起点与训练顺序，
    # 避免后一种配置继承前一实验的参数或优化器状态。
    # Why input statistics come only from training:
    # 预处理属于学习流程；验证数据只用来评价，不参与统计量拟合。
    # 训练集和验证集使用同一组均值/标准差，输入表示才保持一致。
    return histories


if __name__ == "__main__":
    print("--- Example 1: learning rate ---")
    demo_learning_rates()
    print("\n--- Example 2: batch size ---")
    demo_batch_sizes()
    print("\n--- Practice 1: standardization ---")
    practice_standardization()
    print("\n--- Practice 2: mini-batch epoch ---")
    practice_one_minibatch_epoch()
    print("\n--- Homework: controlled experiments ---")
    homework_optimizer_comparison()
