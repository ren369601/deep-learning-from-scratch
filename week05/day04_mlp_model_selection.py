"""Week 5 / Day 4: XOR representations, forward/backward, and model selection."""

import torch
from torch import nn


def demo_fixed_xor():
    """Example: nonlinear hidden features express XOR at four clean corners."""
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0],
                      [1.0, 0.0], [1.0, 1.0]])
    summed = X.sum(dim=1, keepdim=True)
    hidden = torch.cat([torch.relu(summed), torch.relu(summed - 1)], dim=1)
    scores = hidden[:, 0] - 2 * hidden[:, 1]
    print("hidden features:", hidden)
    print("XOR scores:", scores)
    print("predictions:", (scores >= 0.5).long())


def make_xor_data(seed=7):
    """Generate a balanced development set around the four XOR corners."""
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[0.0, 0.0], [0.0, 1.0],
                            [1.0, 0.0], [1.0, 1.0]])
    labels = [0, 1, 1, 0]
    points, targets = [], []
    for center, label in zip(centers, labels):
        points.append(center + 0.08 * torch.randn((12, 2), generator=generator))
        targets.append(torch.full((12,), label, dtype=torch.long))
    return torch.cat(points), torch.cat(targets)


def make_stratified_folds(y, k=3, seed=13):
    """Return validation indices; distribute each shuffled class across folds."""
    generator = torch.Generator().manual_seed(seed)
    parts = [[] for _ in range(k)]
    for label in y.unique():
        indices = torch.where(y == label)[0]
        if len(indices) < k:
            raise ValueError("Each class needs at least k samples.")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        for fold_id, chunk in enumerate(torch.tensor_split(indices, k)):
            parts[fold_id].append(chunk)
    return [torch.cat(chunks) for chunks in parts]


def demo_fold_indices():
    """Example: each fold is held out once and never overlaps its own training."""
    y = torch.tensor([0] * 6 + [1] * 6)
    folds = make_stratified_folds(y)
    for fold_id, val_idx in enumerate(folds):
        train_idx = torch.cat([part for j, part in enumerate(folds) if j != fold_id])
        print("fold:", fold_id, "train:", train_idx.tolist(), "val:", val_idx.tolist())
        print("overlap:", torch.isin(train_idx, val_idx).any().item())


def practice_forward_equivalence():
    """Practice 1: compare manual matrix operations to registered Linear layers."""
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0],
                      [1.0, 0.0], [1.0, 1.0]])
    W1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    b1 = torch.tensor([0.0, -1.0])
    W2 = torch.tensor([[1.0], [-2.0]])
    b2 = torch.tensor([0.0])

    # TODO:
    # 1. Compute hidden = relu(X @ W1 + b1), then output = hidden @ W2 + b2.
    # 2. Create Linear(2,2) -> ReLU -> Linear(2,1).
    # 3. Copy W1.T, b1, W2.T, b2 into the corresponding parameters in no_grad().
    # 4. Compare both outputs using allclose and print their shapes.
    # 5. Explain why removing ReLU makes the two Linear layers affine again.
    print("Practice 1: complete manual/module forward comparison.")
    # 1. Manual forward
    hidden_manual = torch.relu(X @ W1 + b1)
    out_manual = hidden_manual @ W2 + b2
    print("Manual output:\n", out_manual)
    print("Manual output shape:", out_manual.shape)

    # 2. Build nn.Sequential model
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1)
    )
    # 3. Load weights: Linear weight is (out,in), so transpose W1, W2
    with torch.no_grad():
        model[0].weight.copy_(W1.T)
        model[0].bias.copy_(b1)
        model[2].weight.copy_(W2.T)
        model[2].bias.copy_(b2)
    out_model = model(X)
    print("Model output:\n", out_model)
    print("Model output shape:", out_model.shape)

    # 4. Compare
    print("All close?", torch.allclose(out_manual, out_model))

    # 5. Explanation
    print("\nExplanation:")
    print("If we remove ReLU, two Linear layers collapse into a single affine transform:")
    print("X @ W1 @ W2 + (b1 @ W2 + b2), still linear, cannot represent XOR.")


def practice_backward_shapes():
    """Practice 2: run an MLP forward and backward without updating parameters."""
    X, y = make_xor_data()
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))

    # TODO:
    # 1. Compute logits and CrossEntropyLoss; print X/y/logits shapes.
    # 2. Call backward; check each registered parameter has a gradient.
    # 3. Print and check parameter/gradient shapes.
    # 4. Count parameters, and explain backward vs optimizer.step().
    print("Practice 2: complete MLP forward/backward and shape checks.")

    # 1. Forward
    logits = model(X)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("logits shape:", logits.shape)

    # 2. Backward
    loss.backward()

    # 3. Print param & grad shapes
    print("\nParameter and gradient shapes:")
    for name, param in model.named_parameters():
        print(f"{name}: param={param.shape}, grad={param.grad.shape}")

    # 4. Count params and explain
    total_params = sum(p.numel() for p in model.parameters())
    print("\nTotal parameters:", total_params)
    print("Explanation:")
    print("loss.backward() uses autograd to run backprop, compute gradients.")
    print("Backward only computes gradients; optimizer.step() is separate step that updates weights.")


def fit_fold(kind, X_train, y_train, X_val, y_val, seed, epochs=200):
    """Provided training helper: fresh parameters and optimizer on every call."""
    torch.manual_seed(seed)
    if kind == "linear":
        model = nn.Linear(2, 2)
    elif kind == "mlp":
        model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    else:
        raise ValueError("kind must be linear or mlp")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_logits, val_logits = model(X_train), model(X_val)
        return {
            "train_loss": loss_fn(train_logits, y_train).item(),
            "train_acc": (train_logits.argmax(1) == y_train).float().mean().item(),
            "val_loss": loss_fn(val_logits, y_val).item(),
            "val_acc": (val_logits.argmax(1) == y_val).float().mean().item(),
        }


def homework_kfold_comparison():
    """Homework: use one set of folds to compare a linear classifier and MLP."""
    X, y = make_xor_data(seed=7)
    folds = make_stratified_folds(y, k=3, seed=13)

    linear_val_accs = []
    linear_val_losses = []
    mlp_val_accs = []
    mlp_val_losses = []

    # TODO:
    # 1. For each kind ("linear", "mlp"), loop over the same three folds.
    # 2. Use the current fold for validation and concatenate others for training.
    # 3. Check there is no overlap and the folds together cover all samples.
    # 4. Call fit_fold with seed=42+fold_id; print all four returned metrics.
    # 5. Report mean validation loss, mean accuracy, and accuracy std(correction=0).
    # 6. In comments, explain which model this experiment favors, why a fresh
    #    model is needed for each fold, and why these are not final test scores.
    #
    # Results and interpretation:
    # Why fresh model/optimizer:
    # Why validation is not a final test:
    print("Homework: complete the three-fold model comparison.")

    # 1. Loop over folds and model kinds
    for fold_id, val_idx in enumerate(folds):
        # 2. train = all folds except current val fold
        train_idx = torch.cat([f for j, f in enumerate(folds) if j != fold_id])
        # 3. check no overlap and full coverage
        overlap = torch.isin(train_idx, val_idx).any().item()
        all_samples = torch.cat([train_idx, val_idx])
        full_coverage = torch.sort(all_samples)[0].equal(torch.arange(len(y)))
        print(f"\nFold {fold_id}: overlap={overlap}, full_coverage={full_coverage}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        seed = 42 + fold_id

        # train linear
        res_lin = fit_fold("linear", X_train, y_train, X_val, y_val, seed=seed)
        print(f"Linear fold{fold_id}: {res_lin}")
        linear_val_accs.append(res_lin["val_acc"])
        linear_val_losses.append(res_lin["val_loss"])

        # train mlp
        res_mlp = fit_fold("mlp", X_train, y_train, X_val, y_val, seed=seed)
        print(f"MLP fold{fold_id}: {res_mlp}")
        mlp_val_accs.append(res_mlp["val_acc"])
        mlp_val_losses.append(res_mlp["val_loss"])

    # 5. Mean and std (correction=0)
    linear_val_accs = torch.tensor(linear_val_accs)
    linear_val_losses = torch.tensor(linear_val_losses)
    mlp_val_accs = torch.tensor(mlp_val_accs)
    mlp_val_losses = torch.tensor(mlp_val_losses)

    print("\n===== K-Fold Summary =====")
    print(
        f"Linear: mean_val_acc={linear_val_accs.mean().item():.3f}, std={linear_val_accs.std(correction=0).item():.3f}, mean_val_loss={linear_val_losses.mean().item():.3f}")
    print(
        f"MLP:    mean_val_acc={mlp_val_accs.mean().item():.3f}, std={mlp_val_accs.std(correction=0).item():.3f}, mean_val_loss={mlp_val_losses.mean().item():.3f}")

    # 6. Comments
    """
    Results and interpretation:
    The experiment favors MLP. XOR is non-linear separable; linear model can only draw straight decision boundary and cannot fit XOR well.

    Why fresh model/optimizer for each fold:
    Each fold is an independent trial. If reuse old weights, previous fold learned information leaks into next fold, causing biased evaluation. We need each fold start from random initialization.

    Why validation scores are NOT final test scores:
    We use validation folds to select hyperparameters / compare models. Once we pick best model based on validation performance, validation set has information leakage.
    Final generalization performance must be measured on a held-out test set that has never been seen during model comparison or tuning.
    """


if __name__ == "__main__":
    print("--- Example 1: XOR representation ---")
    demo_fixed_xor()
    print("\n--- Example 2: fold indices ---")
    demo_fold_indices()
    print("\n--- Practice 1: equivalent forward ---")
    practice_forward_equivalence()
    print("\n--- Practice 2: backward shapes ---")
    practice_backward_shapes()
    print("\n--- Homework: model selection ---")
    homework_kfold_comparison()
