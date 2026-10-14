"""Week 5 / Day 7: completed reference review of models and experiment practice."""

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from day04_mlp_model_selection import make_stratified_folds, make_xor_data


def demo_output_and_loss():
    """Example: one-logit binary and two-logit multiclass loss formats."""
    binary_logits = torch.tensor([[-1.0], [1.0]])
    binary_targets = torch.tensor([[0.0], [1.0]])
    class_logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    class_targets = torch.tensor([0, 1], dtype=torch.long)
    print("binary BCE:", nn.BCEWithLogitsLoss()(binary_logits, binary_targets).item())
    print("two-class CE:", nn.CrossEntropyLoss()(class_logits, class_targets).item())


def prepare_fold():
    """Use one fixed validation fold; estimate preprocessing on training only."""
    X, y = make_xor_data(seed=7)
    folds = make_stratified_folds(y, k=3, seed=13)
    val_idx = folds[0]
    train_idx = torch.cat(folds[1:])
    assert not torch.isin(train_idx, val_idx).any().item()
    assert torch.equal(torch.cat(folds).sort().values, torch.arange(len(y)))
    mean = X[train_idx].mean(0, keepdim=True)
    std = X[train_idx].std(0, correction=0, keepdim=True).clamp_min(1e-6)
    return (X[train_idx] - mean) / std, y[train_idx], (X[val_idx] - mean) / std, y[val_idx]


def practice_data_and_shapes():
    """Practice 1 solution: check the development split and model dimensions."""
    X_train, y_train, X_val, y_val = prepare_fold()
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    assert model(X_train).shape == (len(y_train), 2)
    assert y_train.dtype == torch.long
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 82
    print("train/validation:", X_train.shape, X_val.shape)
    print("standardized training mean:", X_train.mean(0))
    print("parameters:", count)


def practice_modes_and_checkpoint():
    """Practice 2 solution: preserve and restore independent parameter values."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 2))
    X = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    model.eval()
    with torch.no_grad():
        before = model(X).clone()
    checkpoint = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model[3].bias.add_(1.0)
    model.load_state_dict(checkpoint)
    with torch.no_grad():
        restored = model(X)
    assert torch.equal(before, restored)
    dropout = nn.Dropout(0.5)
    dropout.eval()
    assert torch.equal(dropout(torch.ones(8)), torch.ones(8))
    print("independent checkpoint restored:", torch.equal(before, restored))
    # eval 控制层行为，no_grad 控制求导图；恢复快照则由 load_state_dict 完成。


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        predicted = logits.argmax(1)
        return nn.functional.cross_entropy(logits, y).item(), predicted


def homework_integrated_training():
    """Homework solution: mini-batch MLP, validation, checkpoint, and metrics."""
    X_train, y_train, X_val, y_val = prepare_fold()
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.001)
    generator = torch.Generator().manual_seed(202)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16,
                        shuffle=True, generator=generator)
    best_loss, _ = evaluate(model, X_val, y_val)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience, bad_epochs, updates = 30, 0, 0
    history = []

    for epoch in range(1, 201):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            updates += 1
        train_loss, train_pred = evaluate(model, X_train, y_train)
        val_loss, val_pred = evaluate(model, X_val, y_val)
        train_acc = (train_pred == y_train).float().mean().item()
        val_acc = (val_pred == y_val).float().mean().item()
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            train_acc=train_acc, val_acc=val_acc))
        if epoch in [1, 50, 100, 200]:
            print(f"epoch={epoch}, updates={updates} | train loss={train_loss:.6f}, "
                  f"acc={train_acc:.3f} | val loss={val_loss:.6f}, acc={val_acc:.3f}")
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    final_loss, predicted = evaluate(model, X_val, y_val)
    assert abs(final_loss - best_loss) < 1e-6
    confusion = torch.bincount(2 * y_val + predicted, minlength=4).reshape(2, 2)
    assert confusion.sum().item() == len(y_val)
    final_acc = (predicted == y_val).float().mean().item()
    print("best epoch:", best_epoch, "stop epoch:", epoch)
    print("restored validation loss/accuracy:", final_loss, final_acc)
    print("confusion matrix (rows=true, columns=predicted):", confusion)
    # 数据是开发数据，验证参与早停；这里不是独立测试或三折最终汇总。
    return dict(history=history, best_epoch=best_epoch, stop_epoch=epoch,
                val_loss=final_loss, val_acc=final_acc, confusion=confusion)


if __name__ == "__main__":
    print("--- Example: output and loss ---")
    demo_output_and_loss()
    print("\n--- Practice 1: data and shapes ---")
    practice_data_and_shapes()
    print("\n--- Practice 2: modes and checkpoint ---")
    practice_modes_and_checkpoint()
    print("\n--- Homework: integrated training ---")
    homework_integrated_training()
