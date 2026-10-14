"""Week 5 / Day 6: normalization, dropout, and overfitting experiments."""

import copy

import torch
from torch import nn


def demo_normalization():
    """Example: with rows as samples, BN normalizes columns and LN rows."""
    X = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    batch_norm = nn.BatchNorm1d(2, affine=False)
    layer_norm = nn.LayerNorm(2, elementwise_affine=False)
    batch_norm.train()
    print("BatchNorm training:", batch_norm(X))
    print("LayerNorm:", layer_norm(X))
    batch_norm.eval()
    print("BatchNorm evaluation with running statistics:", batch_norm(X))


def practice_dropout_modes():
    """Practice 1: observe training masks and unchanged evaluation values."""
    torch.manual_seed(42)
    values = torch.ones(20)
    dropout = nn.Dropout(p=0.5)

    # TODO:
    # 1. In train mode, call dropout twice; print both outputs.
    # 2. In eval mode, call twice and check both equal values.
    # 3. Explain the 0/2 values, the random mask, and why eval matters.
    dropout.train()
    train_first = dropout(values)
    train_second = dropout(values)
    print("training output 1:", train_first)
    print("training output 2:", train_second)

    dropout.eval()
    eval_first = dropout(values)
    eval_second = dropout(values)
    print("evaluation output 1:", eval_first)
    print("evaluation output 2:", eval_second)
    assert torch.equal(eval_first, values)
    assert torch.equal(eval_second, values)
    # p=0.5 表示随机丢弃一部分激活；保留的 1 乘以 1/(1-p)，变为 2。
    # 每次训练前向重新抽取掩码，不要求恰好一半元素被丢弃。
    # eval 模式关闭 Dropout，评价不再混入这个随机屏蔽过程。


def practice_early_stopping():
    """Practice 2: count consecutive non-improvements in a fixed loss sequence."""
    losses = [0.8, 0.6, 0.55, 0.56, 0.58, 0.59]
    patience = 2
    min_delta = 0.005

    # TODO:
    # 1. Start best_loss=inf, bad_epochs=0, best_epoch=0.
    # 2. Iterate from epoch 1; reset bad_epochs on sufficient improvement.
    # 3. Otherwise increment bad_epochs; stop when it reaches patience.
    # 4. Print best_epoch and stop_epoch. Explain why they are different.
    best_loss = float("inf")
    bad_epochs = 0
    best_epoch = 0
    stop_epoch = 0
    for epoch, val_loss in enumerate(losses, start=1):
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        stop_epoch = epoch
        print(f"epoch={epoch}, val_loss={val_loss:.3f}, bad_epochs={bad_epochs}")
        if bad_epochs >= patience:
            break
    print("best epoch:", best_epoch, "stop epoch:", stop_epoch)
    assert best_epoch == 3 and stop_epoch == 5
    # 第 3 轮达到最佳记录；第 4、5 轮连续没有足够改善，第 5 轮停止。
    # 真正训练时还要恢复第 3 轮保存的参数，不能直接使用停止时的参数。


def make_noisy_regression(seed=7):
    """Draw separate train/validation samples from the same noisy relationship."""
    generator = torch.Generator().manual_seed(seed)

    def split(count):
        X = 6 * torch.rand((count, 1), generator=generator) - 3
        y = torch.sin(2 * X) + 0.4 * torch.randn((count, 1), generator=generator)
        return X, y

    X_train, y_train = split(16)
    X_val, y_val = split(128)
    return X_train, y_train, X_val, y_val


def evaluate_mse(model, X, y):
    """Provided evaluation always disables Dropout and gradient recording."""
    model.eval()
    with torch.no_grad():
        return nn.functional.mse_loss(model(X), y).item()


def run_regularization(dropout=0.0, weight_decay=0.0, patience=None, epochs=400):
    """Provided trainer: restore the best checkpoint only when early stop is used."""
    X_train, y_train, X_val, y_val = make_noisy_regression()
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(), nn.Dropout(dropout),
        nn.Linear(64, 64), nn.Tanh(), nn.Dropout(dropout), nn.Linear(64, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    initial_train = evaluate_mse(model, X_train, y_train)
    initial_val = evaluate_mse(model, X_val, y_val)
    history = [dict(epoch=0, train_mse=initial_train, val_mse=initial_val)]
    best_loss, best_epoch = initial_val, 0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    min_delta = 1e-4
    stop_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        train_mse = evaluate_mse(model, X_train, y_train)
        val_mse = evaluate_mse(model, X_val, y_val)
        if not torch.isfinite(torch.tensor([train_mse, val_mse])).all().item():
            raise RuntimeError("Non-finite metrics; inspect the training setup.")
        history.append(dict(epoch=epoch, train_mse=train_mse, val_mse=val_mse))
        stop_epoch = epoch
        if val_mse < best_loss - min_delta:
            best_loss, best_epoch = val_mse, epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if patience is not None and bad_epochs >= patience:
            break

    if patience is not None:
        model.load_state_dict(best_state)
    return dict(
        history=history, best_epoch=best_epoch, best_val_mse=best_loss,
        stop_epoch=stop_epoch, final_train_mse=evaluate_mse(model, X_train, y_train),
        final_val_mse=evaluate_mse(model, X_val, y_val),
    )


def homework_regularization_comparison():
    """Homework: compare four independently initialized runs and explain results."""
    configurations = [
        ("baseline", dict()),
        ("dropout", dict(dropout=0.3)),
        ("weight_decay", dict(weight_decay=0.03)),
        ("early_stopping", dict(patience=25)),
    ]

    # TODO:
    # 1. Run each configuration separately with run_regularization(**options).
    # 2. Print metrics at epochs 0, 50, 200, 400 when that epoch was reached.
    # 3. Print best_epoch, stop_epoch, final_train_mse, and final_val_mse.
    # 4. Compare baseline best validation record with its last epoch.
    # 5. For early stopping, compare final_val_mse with best_val_mse.
    # 6. Explain observed tradeoffs, why training MSE is evaluated in eval mode,
    #    and why the selected validation result is not an independent test.
    #
    results = {}
    for name, options in configurations:
        result = run_regularization(**options)
        results[name] = result
        print(f"\nconfiguration: {name}")
        for epoch in [0, 50, 200, 400]:
            if epoch <= result["stop_epoch"]:
                row = result["history"][epoch]
                print(
                    f"epoch={epoch:3d} | train MSE={row['train_mse']:.6f} | "
                    f"val MSE={row['val_mse']:.6f}"
                )
        print(
            f"best epoch={result['best_epoch']}, stop epoch={result['stop_epoch']} | "
            f"selected train MSE={result['final_train_mse']:.6f} | "
            f"selected val MSE={result['final_val_mse']:.6f}"
        )

    baseline = results["baseline"]
    print("\nbaseline best val MSE:", baseline["best_val_mse"])
    print("baseline last val MSE:", baseline["history"][-1]["val_mse"])
    early = results["early_stopping"]
    print("early stopping last val MSE:", early["history"][-1]["val_mse"])
    print("early stopping restored val MSE:", early["final_val_mse"])
    assert abs(early["final_val_mse"] - early["best_val_mse"]) < 1e-6

    # Observations:
    # 本次基线后期训练误差较低，验证误差却高于较早的最佳记录，出现过拟合。
    # Dropout 的最终训练误差较高，验证误差较基线低，体现拟合与泛化的取舍。
    # 当前权重衰减系数只部分缓解问题，不保证每种正则化配置都表现最好。
    # 早停选取较早的模型；结论只适用于本次小数据、种子与配置。
    # Why eval mode for both splits:
    # 用相同的评估行为比较预测误差，避免一边启用 Dropout、另一边关闭。
    # Why restore best checkpoint:
    # 连续无改善才会停止，停止参数不一定是最佳记录对应的参数。
    # deepcopy 保存独立快照，load_state_dict 恢复后再计算最终指标。
    # Why validation is not a final test:
    # 验证结果参与了训练时点和方法的选择，因此还需独立测试集最终评价。
    return results


if __name__ == "__main__":
    print("--- Example: normalization dimensions ---")
    demo_normalization()
    print("\n--- Practice 1: Dropout modes ---")
    practice_dropout_modes()
    print("\n--- Practice 2: early stopping ---")
    practice_early_stopping()
    print("\n--- Homework: regularization experiments ---")
    homework_regularization_comparison()
