"""Week 5 / Day 2: logistic regression, softmax, and a perceptron update."""

import torch
from torch import nn
from torch.optim import SGD



def demo_binary_probabilities():
    """Example: scores become probabilities; loss still takes raw scores."""
    logits = torch.tensor([[-2.0], [0.0], [2.0]])
    targets = torch.tensor([[0.0], [1.0], [1.0]])
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()
    loss = nn.BCEWithLogitsLoss()(logits, targets)
    print("logits:", logits.flatten())
    print("probabilities:", probabilities.flatten())
    print("predictions:", predictions.flatten())
    print("binary loss:", loss.item())


def demo_softmax_and_perceptron():
    """Example: multiclass probabilities, then a separate perceptron update."""
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    targets = torch.tensor([0], dtype=torch.long)
    print("softmax probabilities:", torch.softmax(logits, dim=1))
    print("multiclass loss:", nn.CrossEntropyLoss()(logits, targets).item())

    x = torch.tensor([1.0, 2.0])
    label = -1.0
    weights = torch.tensor([1.0, 0.0])
    bias = 0.0
    learning_rate = 0.1
    score_before = torch.dot(x, weights).item() + bias
    if label * score_before <= 0:
        weights = weights + learning_rate * label * x
        bias = bias + learning_rate * label
    score_after = torch.dot(x, weights).item() + bias
    print("perceptron scores before/after:", score_before, score_after)
    print("perceptron updated weights/bias:", weights, bias)


def practice_binary_forward():
    """Practice 1: follow shapes from two input features to one probability."""
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    weights = torch.tensor([[1.0], [-1.0]])
    bias = 0.0
    targets = torch.tensor([[1.0], [0.0], [1.0]])

    # TODO:
    # 1. Write expected shapes for logits, probabilities, and predictions.
    # 2. Compute logits = X @ weights + bias.
    # 3. Compute sigmoid probabilities and threshold predictions.
    # 4. Compute BCEWithLogitsLoss from logits, then accuracy.
    # 5. Print the values and explain why a zero logit predicts 1 here.
    print("Practice 1: complete binary forward, loss, and accuracy.")
    # 1. Write expected shapes for logits, probabilities, and predictions.
    # logits shape (3,1), probabilities (3,1), predictions (3,1)
    # 2. Compute logits = X @ weights + bias.
    logits = X @ weights + bias
    # 3. Compute sigmoid probabilities and threshold predictions.
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()
    # 4. Compute BCEWithLogitsLoss from logits, then accuracy.
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, targets)
    acc = (predictions == targets).float().mean()
    # 5. Print the values and explain why a zero logit predicts 1 here.
    print("Practice 1: complete binary forward, loss, and accuracy.")
    print("logits:\n", logits)
    print("probabilities:\n", probabilities)
    print("predictions:\n", predictions)
    print("BCE loss:", loss.item())
    print("accuracy:", acc.item())
    # Explanation: logit=0 → sigmoid(0)=0.5, threshold >=0.5 → predict class 1.


def practice_multiclass_forward():
    """Practice 2: normalize each row and use integer class labels."""
    logits = torch.tensor([
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 1.0],
        [1.0, 0.0, 2.0],
    ])
    targets = torch.tensor([0, 1, 2], dtype=torch.long)

    # TODO:
    # 1. Apply softmax along the class dimension; print each row sum.
    # 2. Get predicted classes from logits.argmax(dim=1).
    # 3. Compute CrossEntropyLoss from raw logits, then accuracy.
    # 4. Print shapes and results. Explain why targets has shape (3,).
    print("Practice 2: complete multiclass probabilities and evaluation.")
    # 1. Apply softmax along the class dimension; print each row sum.
    probs = torch.softmax(logits, dim=1)
    print("softmax probs:\n", probs) # 3*3 都是概率
    print("row sums:", probs.sum(dim=1)) #每行加起来都是1
    # 2. Get predicted classes from logits.argmax(dim=1).
    predicted = logits.argmax(dim=1)  # 找每行最大的数，不用softmax也可以找
    # 3. Compute CrossEntropyLoss from raw logits, then accuracy.
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets) #对 logits 做 softmax   计算负对数损失 NLLLoss
    acc = (predicted == targets).float().mean()
    # 4. Print shapes and results. Explain why targets has shape (3,).
    print("Practice 2: complete multiclass probabilities and evaluation.")
    print("logits shape", logits.shape)
    print("predicted classes:", predicted)
    print("cross entropy loss:", loss.item())
    print("accuracy:", acc.item())
    # Explanation: CrossEntropyLoss expects class index labels with shape (N,) not (N,1).


def homework_train_logistic():
    """Homework: train one-output logistic regression and compare a baseline."""
    X_train = torch.tensor([
        [-2.0, -1.0], [-1.0, -2.0], [-1.0, 0.0], [0.0, -1.0],
        [1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, 1.0],
    ])
    y_train = torch.tensor([[0.0], [0.0], [0.0], [0.0],
                            [1.0], [1.0], [1.0], [1.0]])
    X_val = torch.tensor([
        [-1.5, -0.5], [-0.5, -0.5], [0.5, 0.5], [1.5, 0.5],
    ])
    y_val = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

    torch.manual_seed(42)
    model = nn.Linear(2, 1)
    loss_fn = nn.BCEWithLogitsLoss()

    # TODO:
    # 1. Create SGD(model.parameters(), lr=0.1), and train for 200 epochs.
    # 2. Only use X_train/y_train to compute gradients and update parameters.
    # 3. Write an evaluate helper using eval() and no_grad().
    #    Report BCE and accuracy after updates 1, 50, and 200 for both splits.
    # 4. Print final validation probabilities and predicted classes.
    # 5. Select a majority class using ONLY y_train. Report its accuracy
    #    on the same validation set. State your tie-breaking rule.
    # 6. Answer the four questions below in your own words.

    # One logit represents two classes because:
    # BCEWithLogitsLoss does not need a preceding Sigmoid because:
    # Loss can fall while accuracy stays unchanged because:
    # A raw-feature linear classifier cannot solve XOR because:
    print("Homework: complete logistic training and the majority baseline.")
    # 1. Create SGD(model.parameters(), lr=0.1), and train for 200 epochs.
    opt = SGD(model.parameters(), lr=0.1)

    # 3. Write an evaluate helper using eval() and no_grad().
    def evaluate(model, X, y):
        model.eval()
        with torch.no_grad():
            logits = model(X)
            loss = loss_fn(logits, y)
            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).float()
            acc = (pred == y).float().mean()
        return loss.item(), acc.item(), probs, pred

    # 2. Only use X_train/y_train to compute gradients and update parameters.
    epochs = 200
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

        # Report at epoch 1,50,200
        if epoch in [1, 50, 200]:
            tr_loss, tr_acc, _, _ = evaluate(model, X_train, y_train)
            v_loss, v_acc, _, _ = evaluate(model, X_val, y_val)
            print(f"\nEpoch {epoch}")
            print(f"Train BCE: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}")
            print(f"Val   BCE: {v_loss:.4f}, Val   Acc: {v_acc:.4f}")

    # 4. Print final validation probabilities and predicted classes.
    val_loss, val_acc, val_probs, val_pred = evaluate(model, X_val, y_val)
    print("\nFinal validation probabilities:\n", val_probs) # 类别一的概率
    print("Final validation predictions:\n", val_pred)

    # 5. Select a majority class using ONLY y_train. Report its accuracy
    #    on the same validation set. State your tie-breaking rule.
    # y_train: 4 zeros,4 ones → tie; tie break: pick class 0
    majority_class = 0.0
    majority_pred = torch.full_like(y_val, majority_class)
    majority_acc = (majority_pred == y_val).float().mean()
    print(f"\nMajority baseline (tie break to 0): val accuracy = {majority_acc.item()}")

    # 6. Answer the four questions below in your own words.
    # One logit represents two classes because:
    # BCEWithLogitsLoss does not need a preceding Sigmoid because:
    # Loss can fall while accuracy stays unchanged because:
    # A raw-feature linear classifier cannot solve XOR because:
    print("\nHomework: complete logistic training and the majority baseline.")
    print("""
    Answers:
    1. One logit represents two classes because:
    The single raw score measures evidence for positive class.
    Sigmoid converts logit to P(y=1). P(y=0)=1-P(y=1).

    2. BCEWithLogitsLoss does not need a preceding Sigmoid because:
    It internally computes sigmoid + binary cross entropy in one stable operation.

    3. Loss can fall while accuracy stays unchanged because:
    Loss measures confidence, accuracy only counts hard 0/1 decisions.
    The model can get more confident (lower loss) without crossing the 0.5 threshold.

    4. A raw-feature linear classifier cannot solve XOR because:
    XOR data cannot be separated by a single straight line (linearly inseparable).
    """)

if __name__ == "__main__":
    print("--- Example 1: binary probabilities ---")
    demo_binary_probabilities()

    print("\n--- Example 2: softmax and perceptron ---")
    demo_softmax_and_perceptron()

    print("\n--- Practice 1: binary forward ---")
    practice_binary_forward()

    print("\n--- Practice 2: multiclass forward ---")
    practice_multiclass_forward()

    print("\n--- Homework: logistic baseline ---")
    homework_train_logistic()
