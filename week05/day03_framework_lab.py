"""Week 5 / Day 3: NumPy conversion, two regression implementations, metrics."""

import numpy as np
import torch
from torch import nn


def demo_conversion_and_autograd():
    """Example: shared storage, a copy, and one automatically computed gradient."""
    array = np.array([1.0, 2.0], dtype=np.float32)
    shared = torch.from_numpy(array)
    copied = torch.tensor(array)
    array[0] = 9.0
    print("NumPy array:", array)
    print("shared tensor:", shared)
    print("copied tensor:", copied)

    weight = torch.tensor(2.0, requires_grad=True)
    loss = (3 * weight - 9) ** 2
    loss.backward()
    print("gradient:", weight.grad.item())
    print("NumPy snapshot:", weight.detach().cpu().numpy().copy())
    print("CUDA available:", torch.cuda.is_available())


def make_regression_data():
    """Provide the same CPU float32 data to both training implementations."""
    X_train = torch.tensor([
        [-2.0, 0.0], [-1.0, 1.0], [0.0, -1.0],
        [1.0, 0.0], [2.0, 1.0], [1.0, -2.0],
    ])
    y_train = 2 * X_train[:, :1] - X_train[:, 1:] + 1
    X_val = torch.tensor([[-1.5, 0.5], [0.5, -0.5], [0.0, 0.5]])
    y_val = 2 * X_val[:, :1] - X_val[:, 1:] + 1
    return X_train, y_train, X_val, y_val


def practice_manual_regression():
    """Practice 1: let Autograd find gradients; manage parameters yourself."""
    X_train, y_train, X_val, y_val = make_regression_data()
    W = torch.zeros((2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    loss_fn = nn.MSELoss()

    # TODO:
    # 1. Train for 300 epochs with learning_rate=0.05.
    # 2. Forward -> MSE -> backward, then update W and b inside no_grad().
    # 3. Clear both parameter gradients before the next backward.
    # 4. Re-evaluate train MSE and val MAE using the final parameters.
    # 5. Print W, b, and those metrics. Expected shapes: W (2,1), b (1,).
    # 6. Explain why "manual parameters" does not mean "manual backward".
    print("Practice 1: complete manual-parameter regression.")
    for epoch in range(300):
        # 1.前向
        pred = X_train @ W + b
        loss = loss_fn(pred, y_train)
        # 2.反向求梯度
        loss.backward()

        # 3.更新参数，必须包no_grad()
        with torch.no_grad():
            W -= 0.05 * W.grad
            b -= 0.05 * b.grad

        # 4.梯度清零！！
        W.grad.zero_()
        b.grad.zero_()

    # 训练结束，在no_grad下评估训练MSE、验证MAE
    with torch.no_grad():
        pred_train = X_train @ W + b
        train_mse = loss_fn(pred_train, y_train)
        pred_val = X_val @ W + b
        val_mae = torch.mean(torch.abs(pred_val - y_val))

    print("W:\n", W, "\nb:", b)
    print("train MSE:", train_mse.item(), "val MAE:", val_mae.item())




class LinearRegressor(nn.Module):
    """Practice 2: register a two-feature, one-output Linear layer."""

    def __init__(self):
        super().__init__()
        # TODO: Register self.linear = nn.Linear(2, 1).
        self.linear = nn.Linear(2, 1)  # 输入2特征，输出1

    def forward(self, X):
        # TODO: Return self.linear(X).
        return self.linear(X)


def practice_module_regression():
    """Practice 2: repeat the same experiment with nn.Module and an optimizer."""
    X_train, y_train, X_val, y_val = make_regression_data()

    # TODO:
    # 1. Complete LinearRegressor, then instantiate it.
    # 2. Zero its weight and bias inside no_grad() to match Practice 1.
    # 3. Use MSELoss and SGD(lr=0.05) for the same 300 epochs.
    # 4. Re-evaluate train MSE and val MAE with eval() and no_grad().
    # 5. Print linear.weight.T and linear.bias; compare with Practice 1.
    # 6. Explain why the same task needs the same data and initialization
    #    for a useful numerical comparison.
    print("Practice 2: complete nn.Module regression and the comparison.")

    model = LinearRegressor() # 随机初始化WB
    # 关键：把weight、bias手动置0，和practice1初始化完全一样，公平对比
    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.bias.fill_(0.0)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(300):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()  # 自动更新所有参数，不用手写W,b更新

    # 评估，加model.eval()和no_grad
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train)
        train_mse = loss_fn(pred_train, y_train)
        pred_val = model(X_val)
        val_mae = torch.mean(torch.abs(pred_val - y_val))
    # 打印的时候 .T转置！因为nn.Linear weight shape(1,2)，转置后和practice1 W(2,1)对比
    print("linear.weight.T:\n", model.linear.weight.T, "\nbias:", model.linear.bias)
    print("train MSE:", train_mse.item(), "val MAE:", val_mae.item())



def homework_classification_metrics():
    """Homework: evaluate fixed binary scores; no extra model training required."""
    logits = torch.tensor([[-2.0], [1.0], [-1.0], [-0.4],
                           [-0.8], [0.5], [2.0], [3.0]])
    targets = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

    # TODO:
    # 1. Convert sigmoid probabilities to classes using threshold 0.5.
    #    Flatten predictions to (8,) so they match targets.
    # 2. Compute TN, FP, FN, TP with Boolean conditions and sums.
    # 3. Build [[TN, FP], [FN, TP]]; print the row/column convention.
    # 4. Compute accuracy, precision, recall, and F1.
    #    Return 0.0 for any metric with a zero denominator.
    # 5. Check that the four counts sum to the number of examples.
    # 6. In comments, explain precision vs recall and why accuracy alone
    #    can be misleading when one class is rare.
    #
    # Precision means:
    # Recall means:
    # An accuracy-only limitation:
    print("Homework: complete confusion matrix and classification metrics.")

    # 1. Convert sigmoid probabilities to classes using threshold 0.5.
    #    Flatten predictions to (8,) so they match targets.
    prob = torch.sigmoid(logits)
    pred = (prob >= 0.5).long()
    pred = pred.flatten()  # shape (8,)

    # 2. Compute TN, FP, FN, TP with Boolean conditions and sums.
    y_true = targets
    TP = torch.sum((pred == 1) & (y_true == 1)).item()
    TN = torch.sum((pred == 0) & (y_true == 0)).item()
    FP = torch.sum((pred == 1) & (y_true == 0)).item()
    FN = torch.sum((pred == 0) & (y_true == 1)).item()

    # 3. Build [[TN, FP], [FN, TP]]; print the row/column convention.
    confusion = torch.tensor([[TN, FP],
                              [FN, TP]])
    print("Confusion matrix [[TN,FP],[FN,TP]]:")
    print(confusion)
    print("Row: true label; Column: predicted label")

    # 4. Compute accuracy, precision, recall, and F1.
    #    Return 0.0 for any metric with a zero denominator.
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")

    # 5. Check that the four counts sum to the number of examples.
    assert TP + TN + FP + FN == len(targets), "count mismatch"

    # 6. In comments, explain precision vs recall and why accuracy alone
    #    can be misleading when one class is rare.
    #
    # Precision means:
    # 所有预测为正类的样本中，真实是正样本的比例。衡量预测出来的正样本“靠不靠谱”。
    # Recall means:
    # 所有真实的正样本里，被模型成功检测出来的比例。衡量模型能不能把正样本找全。
    # An accuracy-only limitation:
    # 样本不均衡时（比如负样本极多、正样本很少），无脑预测全部为负也能得到很高准确率，但完全识别不出少数类，准确率无法反映模型对少数类的识别能力。

    print("Homework: complete confusion matrix and classification metrics.")


if __name__ == "__main__":
    print("--- Example: conversion and Autograd ---")
    demo_conversion_and_autograd()

    print("\n--- Practice 1: manual parameters ---")
    practice_manual_regression()

    print("\n--- Practice 2: nn.Module ---")
    practice_module_regression()

    print("\n--- Homework: classification metrics ---")
    homework_classification_metrics()
