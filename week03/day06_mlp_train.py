"""Week 3 / Day 6: train the Day 5 NumPy MLP with backpropagation."""

import numpy as np

from day05_mlp_forward import TwoLayerMLP


def cross_entropy(probabilities, targets):
    """Return mean Cross Entropy for integer class labels."""
    if probabilities.ndim != 2 or targets.ndim != 1:
        raise ValueError("probabilities 必须是二维，targets 必须是一维")
    if probabilities.shape[0] != len(targets):
        raise ValueError("样本数不匹配")

    rows = np.arange(len(targets))
    correct = np.clip(probabilities[rows, targets], 1e-12, 1.0)
    return float(-np.log(correct).mean())


def accuracy(probabilities, targets):
    """Return the fraction of correctly classified samples."""
    return float(np.mean(probabilities.argmax(axis=1) == targets))


def softmax_cross_entropy_gradient(probabilities, targets):
    """Return dLoss/dLogits for mean Cross Entropy."""
    rows = np.arange(len(targets))
    grad_logits = probabilities.copy()
    grad_logits[rows, targets] -= 1.0
    grad_logits /= len(targets)
    return grad_logits


def backward(model, targets):
    """Return gradients for W1, b1, W2, and b2 after model.forward()."""
    # TODO: Use model.probabilities and the cached values from Day 5.
    # 1. dlogits = softmax_cross_entropy_gradient(...).
    dlogits = softmax_cross_entropy_gradient(model.probabilities, targets)
    # 2. Get hidden from model.linear2.last_input.
    hidden = model.linear2.last_input  # A1, 经过ReLU后的输出
    # 3. dW2 = hidden.T @ dlogits; db2 = dlogits.sum(axis=0).
    dW2 = hidden.T @ dlogits
    db2 = dlogits.sum(axis=0)
    # 4. dhidden = dlogits @ model.linear2.weights.T.
    dhidden = dlogits @ model.linear2.weights.T
    # 5. Get hidden_linear from model.relu.last_input.
    hidden_linear = model.relu.last_input  # Z1，ReLU输入
    # 6. dhidden_linear = dhidden * (hidden_linear > 0).
    dhidden_linear = dhidden * (hidden_linear > 0)
    # 7. Get X from model.linear1.last_input.
    X = model.linear1.last_input
    # 8. dW1 = X.T @ dhidden_linear; db1 = dhidden_linear.sum(axis=0).
    dW1 = X.T @ dhidden_linear
    db1 = dhidden_linear.sum(axis=0)
    # 9. Return a dict with keys "W1", "b1", "W2", "b2".
    grads = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }
    return grads


def apply_gradients(model, gradients, learning_rate):
    """Update all parameters after every gradient has been computed."""
    # TODO: Subtract learning_rate * the corresponding gradient from
    # model.linear1.weights, model.linear1.bias,
    # model.linear2.weights, and model.linear2.bias.
    # W = W - lr * grad
    model.linear1.weights -= learning_rate * gradients["W1"]
    model.linear1.bias -= learning_rate * gradients["b1"]
    model.linear2.weights -= learning_rate * gradients["W2"]
    model.linear2.bias -= learning_rate * gradients["b2"]


def train(model, X, targets, learning_rate, epochs, log_every=200):
    """Train on the whole dataset and return loss history."""
    losses = []

    # TODO: For each epoch:
    for epoch in range(epochs):
        # 1. probabilities = model.forward(X).
        prob = model.forward(X)
        # 2. Calculate and append Cross Entropy for the current parameters.
        loss = cross_entropy(prob, targets)
        losses.append(loss)
        # 3. Calculate all gradients with backward(model, targets).
        grads = backward(model, targets)
        # 4. Update parameters with apply_gradients(...).
        apply_gradients(model, grads, learning_rate)
        # 5. Every log_every epochs, print epoch, loss, and accuracy.
        if epoch % log_every == 0:
            acc = accuracy(prob, targets)
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc:.4f}")
        # Return losses.
    return losses


def demo_output_gradient():
    """Show the first backpropagation gradient and its shape."""
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    targets = np.array([0, 1, 1, 0])
    model = TwoLayerMLP(2, 8, 2, seed=42)
    probabilities = model.forward(X)
    grad_logits = softmax_cross_entropy_gradient(probabilities, targets)

    print("probabilities shape:", probabilities.shape)
    print("targets shape:", targets.shape)
    print("initial loss:", cross_entropy(probabilities, targets))
    print("dlogits shape:", grad_logits.shape)
    print("dlogits:\n", grad_logits)
    print("W2 shape:", model.linear2.weights.shape)
    print("expected dW2 shape:", model.linear2.weights.shape)


def practice_gradient_shapes():
    """练习 1：完成 Backward 并核对四组梯度的 shape。"""
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    targets = np.array([0, 1, 1, 0])
    model = TwoLayerMLP(2, 8, 2, seed=42)

    # TODO:
    # 1. Call model.forward(X), then backward(model, targets).
    model.forward(X)
    grads = backward(model, targets)
    # 2. Print each gradient's shape beside its parameter's shape.
    print("W1 param shape:", model.linear1.weights.shape, " | grad W1 shape:", grads["W1"].shape)
    print("b1 param shape:", model.linear1.bias.shape, " | grad b1 shape:", grads["b1"].shape)
    print("W2 param shape:", model.linear2.weights.shape, " | grad W2 shape:", grads["W2"].shape)
    print("b2 param shape:", model.linear2.bias.shape, " | grad b2 shape:", grads["b2"].shape)
    # 3. Confirm all four pairs match.


def practice_one_update():
    """练习 2：观察一次更新前后的 loss。"""
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    targets = np.array([0, 1, 1, 0])
    model = TwoLayerMLP(2, 8, 2, seed=42)

    # TODO:
    # 1. Forward and calculate loss_before.
    prob_before = model.forward(X)
    loss_before = cross_entropy(prob_before, targets)
    # 2. Calculate gradients with backward().
    grads = backward(model, targets)
    # 3. Update all parameters with learning_rate=0.1.
    apply_gradients(model, grads, learning_rate=0.1)
    # 4. Forward again and calculate loss_after.
    prob_after = model.forward(X)
    loss_after = cross_entropy(prob_after, targets)
    # 5. Print both losses
    print(f"loss before update: {loss_before:.4f}")
    print(f"loss after update:  {loss_after:.4f}")
    print("说明：学习率太大，单次更新loss有可能上升；学习率合适才会下降。")


def homework_train_xor():
    """作业：训练一个两层 MLP 学习 XOR 分类。"""
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    targets = np.array([0, 1, 1, 0])
    model = TwoLayerMLP(2, 8, 2, seed=42)

    # TODO:
    # 1. Calculate initial loss and accuracy.
    prob_init = model.forward(X)
    loss_init = cross_entropy(prob_init, targets)
    acc_init = accuracy(prob_init, targets)
    print(f"Initial loss: {loss_init:.4f}, Initial acc: {acc_init:.4f}")
    # 2. Train for about 2000-4000 epochs; try learning_rate=0.1 first.
    loss_history = train(model, X, targets, learning_rate=0.1, epochs=3000, log_every=300)
    # 3. Run a fresh forward and print final loss, accuracy, probabilities, predicted classes, and targets.
    prob_final = model.forward(X)
    loss_final = cross_entropy(prob_final, targets)
    acc_final = accuracy(prob_final, targets)
    pred_classes = prob_final.argmax(axis=1)
    print("\n==== Final Result ====")
    print(f"Final loss: {loss_final:.4f}, Final acc: {acc_final:.4f}")
    print("probabilities:\n", prob_final)
    print("predicted classes:", pred_classes)
    print("targets:          ", targets)
    # 4. Print the first and last entries in the returned loss history.
    print(f"\nFirst loss in history: {loss_history[0]:.4f}")
    print(f"Last loss in history:  {loss_history[-1]:.4f}")
    # 5. If training stalls, inspect gradients and try a different learning rate or seed; record what changed.


if __name__ == "__main__":
    print("--- 输出梯度示例 ---")
    demo_output_gradient()

    print("\n--- 练习 1：梯度 shape ---")
    practice_gradient_shapes()

    print("\n--- 练习 2：一次更新 ---")
    practice_one_update()

    print("\n--- 作业：训练 XOR ---")
    homework_train_xor()
