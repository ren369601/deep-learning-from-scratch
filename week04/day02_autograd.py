"""Week 4 / Day 2: compare PyTorch Autograd with manual gradients."""

import torch


def demo_scalar_gradient():
    """Compare a chain-rule calculation with Autograd."""
    x = torch.tensor(2.0)
    w = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    target = torch.tensor(9.0)

    prediction = x * w + b
    loss = (prediction - target) ** 2
    loss.backward()

    print("prediction:", prediction.item(), "loss:", loss.item())
    print("w.grad:", w.grad.item(), "b.grad:", b.grad.item())
    print("manual dw:", 2 * (prediction.item() - target.item()) * x.item())
    print("manual db:", 2 * (prediction.item() - target.item()))


def practice_scalar_gradient():
    """练习 1：手算并核对两个参数的梯度。"""
    x = torch.tensor(3.0)
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(-1.0, requires_grad=True)
    target = torch.tensor(4.0)

    # TODO:
    # 1.前向计算
    prediction = x * w + b
    loss = (prediction - target) ** 2
    # 2.手算
    pred_val = prediction.item()
    target_val = target.item()
    dw_manual = 2 * (pred_val - target_val) * x.item()
    db_manual = 2 * (pred_val - target_val)
    print(f"手算 loss = {loss.item()}")
    print(f"手算 dw = {dw_manual}, db = {db_manual}")

    # 3. pytorch自动求导
    loss.backward()
    print(f"autograd w.grad={w.grad.item()}, b.grad={b.grad.item()}")

    # 4.核对
    close_w = torch.allclose(torch.tensor(w.grad.item()), torch.tensor(dw_manual))
    close_b = torch.allclose(torch.tensor(b.grad.item()), torch.tensor(db_manual))
    print(f"w梯度是否相等:{close_w}, b梯度是否相等:{close_b}")


def practice_gradient_accumulation():
    """练习 2：观察梯度累积并清零。"""
    x = torch.tensor(3.0)
    w = torch.tensor(2.0, requires_grad=True)
    target = torch.tensor(10.0)

    # TODO:
    # 1. 计算 loss = (x*w-target)**2，backward 后打印 w.grad。
    # 2. 重新计算 prediction 和 loss，再 backward，打印累积的 w.grad。
    # 3. 调用 w.grad.zero_()，重新前向和 backward，打印新的 w.grad。
    # 4. 写一句话说明三次打印的区别。

    # 1 第一次前向+反向
    pred1 = x * w
    loss1 = (pred1 - target) ** 2
    loss1.backward()
    print("第1次 backward, w.grad =", w.grad.item())

    # 2 不重置，第二次backward
    pred2 = x * w
    loss2 = (pred2 - target) ** 2
    loss2.backward()
    print("第2次 backward(不清零), w.grad =", w.grad.item())

    # 3 清零再重新计算
    w.grad.zero_()
    pred3 = x * w
    loss3 = (pred3 - target) ** 2
    loss3.backward()
    print("清零后重新backward, w.grad =", w.grad.item())

    # 4 文字说明
    print("说明：默认梯度会累加；不清零两次反向，梯度叠加；zero_()把旧梯度清零，重新计算本轮梯度。")


def homework_batch_linear_gradients():
    """作业：比较一批数据的自动梯度和手写梯度。"""
    X = torch.tensor([[1.0, 2.0], [2.0, 0.0], [0.0, 1.0]])
    W = torch.tensor([[1.0], [2.0]], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    target = torch.tensor([[6.0], [1.0], [2.0]])

    # TODO:
    # 1. prediction = X @ W + b，loss = ((prediction-target)**2).mean()。
    # 2. 调用 loss.backward()，打印 prediction、loss、W.grad、b.grad 和 shape。
    # 3. 手写 d_prediction = 2*(prediction-target)/prediction.numel()。
    # 4. 手写 dW = X.T @ d_prediction；db = d_prediction.sum(dim=0)。
    # 5. 用 torch.allclose() 检查 W.grad 与 dW、b.grad 与 db。
    # 6. 解释为什么这里要除以 prediction.numel()。

    # 1.前向
    prediction = X @ W + b
    loss = ((prediction - target) ** 2).mean()

    # 2.autograd
    loss.backward()
    print("prediction:\n", prediction)
    print("loss:", loss.item())
    print("W.grad:\n", W.grad, "shape", W.grad.shape)
    print("b.grad:\n", b.grad, "shape", b.grad.shape)

    # 3.手写梯度公式
    d_prediction = 2 * (prediction - target) / prediction.numel()
    # 4
    dW = X.T @ d_prediction
    db = d_prediction.sum(dim=0)
    print("\n手写dW:\n", dW)
    print("手写db:\n", db)

    # 5 校验
    check_W = torch.allclose(W.grad, dW)
    check_b = torch.allclose(b.grad, db)
    print(f"\nW梯度是否匹配:{check_W}, b梯度是否匹配:{check_b}")

    # 6 解释除以numel()
    print(
        "\n解释：loss使用.mean()，是全部元素求平均。求导时平均操作等价于除以总元素数量numel()，所以d_prediction要除以prediction.numel()。")


if __name__ == "__main__":
    print("--- 自动求导示例 ---")
    demo_scalar_gradient()

    print("\n--- 练习 1：标量梯度 ---")
    practice_scalar_gradient()

    print("\n--- 练习 2：梯度累积 ---")
    practice_gradient_accumulation()

    print("\n--- 作业：批量线性层梯度 ---")
    homework_batch_linear_gradients()
