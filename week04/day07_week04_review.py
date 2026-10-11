"""Week 4 / Day 7: review tensors, gradients, and train/validation."""

import torch
from torch import nn

from day05_pytorch_mlp import TwoLayerMLP
from day06_train_validate import make_xor_data


def demo_tiny_autograd():
    """Example: one scalar parameter, one loss, one gradient."""
    weight = torch.tensor(2.0, requires_grad=True)
    prediction = weight * torch.tensor(3.0)
    loss = (prediction - 9.0) ** 2
    loss.backward()
    print("weight:", weight.item(), "loss:", loss.item())
    print("weight.grad:", weight.grad.item())


def practice_shapes_and_parameters():
    """Practice 1: connect input, logits, labels, and parameter shapes."""
    X = torch.zeros((20, 2))
    targets = torch.zeros(20, dtype=torch.long)
    model = TwoLayerMLP(2, 16, 2)

    # TODO:
    # 1. Get logits and print the shapes of X, logits, and targets.
    # 2. Print the shape of each registered parameter.
    # 3. Calculate the total parameter count; write your prediction first.
    # 4. Check that logits and targets have the shapes CrossEntropyLoss needs.
    # 1. 前向拿到logits
    logits = model(X)
    print("X.shape:", X.shape)
    print("logits.shape:", logits.shape)
    print("targets.shape:", targets.shape)

    # 2. 打印每一层参数shape
    for name, param in model.named_parameters():
        print(name, param.shape)

    # 3. 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("total params:", total_params)
    # 预测值先写注释：预测总参数=82

    # 4. CrossEntropy形状校验
    assert logits.ndim == 2
    assert targets.ndim == 1
    assert logits.shape[0] == targets.shape[0]
    print("shape check passed")


def practice_gradient_accumulation():
    """Practice 2: observe what happens when gradients are not cleared."""
    X = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([1, 1], dtype=torch.long)
    torch.manual_seed(42)
    model = TwoLayerMLP(2, 8, 2)
    loss_fn = nn.CrossEntropyLoss()

    # TODO:
    # 1. Compute loss and backward once; clone one parameter's gradient.
    # 2. Compute the same loss and backward again without zero_grad().
    # 3. Compare the selected gradient before and after the second backward.
    # 4. Clear gradients and repeat once; explain what changed.
    # Do not call optimizer.step() in this practice.
    # 1. 第一次前向+反向，克隆梯度
    logits = model(X)
    loss = loss_fn(logits, targets)
    loss.backward()
    # 取第一层bias的梯度做观察
    grad_first = model.linear1.bias.grad.clone()
    print("第一次梯度：", grad_first)

    # 2. 不清零，再来一次 forward backward
    logits2 = model(X)
    loss2 = loss_fn(logits2, targets)
    loss2.backward()
    grad_second = model.linear1.bias.grad
    print("不清零，第二次梯度：", grad_second)
    # 第二次梯度≈2*grad_first

    # 3. 清零梯度，再跑一次
    model.zero_grad()
    logits3 = model(X)
    loss3 = loss_fn(logits3, targets)
    loss3.backward()
    grad_third = model.linear1.bias.grad
    print("清零后，第三次梯度：", grad_third)
    # 清零后梯度回到单次大小，不再累加


def homework_train_validate():
    """Homework: rebuild a small XOR training and validation workflow."""
    X_train, y_train, X_val, y_val = make_xor_data()
    torch.manual_seed(42)
    model = TwoLayerMLP(2, 16, 2)
    loss_fn = nn.CrossEntropyLoss()

    # TODO:
    # 1. Create an Adam optimizer with lr=0.03.
    # 2. Train for 100 epochs using only X_train/y_train for updates.
    # 3. After epochs 1, 50, and 100, report train and val loss/accuracy
    #    using the newly updated model. Write your own evaluate helper.
    # 4. Check that evaluation does not change model parameters.
    # 5. Explain in review.md why these 20 validation examples do not prove
    #    reliable predictions far away from the four XOR neighborhoods.
    # 1. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    # 自己写evaluate
    def evaluate(model, X, targets):
        model.eval()
        with torch.no_grad():
            logits = model(X)
            loss = loss_fn(logits, targets).item()
            acc = (logits.argmax(dim=1) == targets).float().mean().item()
        return loss, acc

    epochs = 100
    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        # 指定epoch打印
        if epoch in [1, 50, 100]:
            tr_loss, tr_acc = evaluate(model, X_train, y_train)
            v_loss, v_acc = evaluate(model, X_val, y_val)
            print(
                f"Epoch {epoch} | train loss:{tr_loss:.4f}, acc:{tr_acc:.2f} | val loss:{v_loss:.4f}, acc:{v_acc:.2f}")

    # 4. 检查评估不会修改参数
    old_params = [p.clone() for p in model.parameters()]
    evaluate(model, X_val, y_val)
    all_same = all(torch.equal(o, n) for o, n in zip(old_params, model.parameters()))
    print("评估后参数是否不变：", all_same)

    # 5. 解释写在注释
    # 验证集的20个样本全部是在4个XOR中心点附近采样，属于局部分布。
    # 模型只学习了这个局部区域的规律；如果输入远离4个角，不属于训练数据分布，
    # 模型从未见过这类样本，预测不可靠，所以不能证明在任意远处数据上都能可靠预测。


if __name__ == "__main__":
    print("--- Example: Autograd ---")
    demo_tiny_autograd()

    print("\n--- Practice 1: shapes and parameters ---")
    practice_shapes_and_parameters()

    print("\n--- Practice 2: gradient accumulation ---")
    practice_gradient_accumulation()

    print("\n--- Homework: train and validate ---")
    homework_train_validate()
