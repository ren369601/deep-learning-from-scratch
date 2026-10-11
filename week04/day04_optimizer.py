"""Week 4 / Day 4: SGD, Adam, and the optimizer training order."""

import torch
from torch import nn


def make_zero_linear():
    """Create a one-input, one-output linear model with fixed starting values."""
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)
    return model


def demo_one_sgd_step():
    """Compare one SGD step with the manual result."""
    model = make_zero_linear()
    X = torch.tensor([[1.0]])
    target = torch.tensor([[3.0]])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    loss_before = loss_fn(model(X), target)
    loss_before.backward()
    print("before: weight=", model.weight.item(), "bias=", model.bias.item())
    print("loss=", loss_before.item(), "dw=", model.weight.grad.item(), "db=", model.bias.grad.item())
    optimizer.step()

    with torch.no_grad():
        loss_after = loss_fn(model(X), target)
    print("after:  weight=", model.weight.item(), "bias=", model.bias.item())
    print("new loss=", loss_after.item())


def practice_one_update():
    """练习 1：在两个样本上完成一次 SGD 更新。"""
    model = make_zero_linear()
    X = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([[3.0], [5.0]])
    loss_fn = nn.MSELoss()

    # TODO:
    # 1. 创建SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # 2. 标准顺序 zero_grad -> forward -> loss -> backward -> step
    optimizer.zero_grad()
    pred = model(X)
    loss_before = loss_fn(pred, target)
    loss_before.backward()

    # 3.打印更新前loss、梯度
    print("练习1更新前:")
    print("loss =", loss_before.item())
    print("dw =", model.weight.grad.item(), "shape", model.weight.grad.shape)
    print("db =", model.bias.grad.item(), "shape", model.bias.grad.shape)

    # 执行SGD更新
    optimizer.step()

    # 4.更新后，重新计算loss
    with torch.no_grad():
        pred_new = model(X)
        loss_after = loss_fn(pred_new, target)
    print("\n练习1更新后:")
    print("weight =", model.weight.item(), "bias =", model.bias.item())
    print("new loss =", loss_after.item())

    # 5.手算推导
    # 模型 y = w*x + b，初始w=0,b=0
    # 样本1：x=1 target=3 → y=0，误差=-3
    # 样本2：x=2 target=5 → y=0，误差=-5
    # MSE = ((-3)^2 + (-5)^2)/2 = (9+25)/2 = 17
    # dL/dw = (2*(y1-t1)*x1 + 2*(y2-t2)*x2)/2 = (-6*1 + -10*2)/2 = (-26)/2 = -13
    # dL/db = (2*(y1-t1) + 2*(y2-t2))/2 = (-6 + -10)/2 = -8
    # w_new = 0 - 0.05*(-13) = 0.65
    # b_new = 0 -0.05*(-8) = 0.4
    # 更新后w=0.65 b=0.4
    # y1=0.65*1+0.4=1.05；y2=0.65*2+0.4=1.7
    # MSE = ((1.05-3)^2 + (1.7-5)^2)/2 = (3.8025 + 10.89)/2 =7.346
    # loss从17下降到7.346，参数往正确方向移动


def practice_training_loop():
    """练习 2：写一个短训练循环。"""
    model = make_zero_linear()
    X = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    target = 2 * X + 1
    loss_fn = nn.MSELoss()

    # TODO:
    # 1. 创建SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)  # 更新权重
    loss_history = []

    # 2.训练30轮
    for epoch in range(30):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, target)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()

    # 3.打印第1、30轮loss
    print("\n练习2：")
    print("第1轮loss:", loss_history[0])
    print("第30轮loss:", loss_history[-1])

    # 4.训练结束，no_grad下前向
    with torch.no_grad():
        final_pred = model(X)
        final_loss = loss_fn(final_pred, target)
    print("训练结束后重新计算的final loss:", final_loss.item())

    # 5.解释：loss_history[-1]保存的是step**之前**算出来的loss
    # 保存loss的时候，参数还没更新；final_loss是参数更新完之后算的，所以数值有微小差别


def homework_compare_optimizers():
    """作业：让 SGD 和 Adam 从相同参数出发。"""
    X = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    target = 2 * X + 1
    loss_fn = nn.MSELoss()
    model_sgd = make_zero_linear()
    model_adam = make_zero_linear()

    # TODO:
    # 1.相同初值，数据，lr=0.05，30轮
    lr = 0.05
    opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=lr)
    opt_adam = torch.optim.Adam(model_adam.parameters(), lr=lr)

    loss_sgd = []
    loss_adam = []
    for _ in range(30):
        # SGD训练一步
        opt_sgd.zero_grad()
        p_sgd = model_sgd(X)
        l_sgd = loss_fn(p_sgd, target)
        loss_sgd.append(l_sgd.item())
        l_sgd.backward()
        opt_sgd.step()

        # Adam训练一步
        opt_adam.zero_grad()
        p_adam = model_adam(X)
        l_adam = loss_fn(p_adam, target)
        loss_adam.append(l_adam.item())
        l_adam.backward()
        opt_adam.step()

    # 测试
    with torch.no_grad():
        pred_sgd = model_sgd(X)
        pred_adam = model_adam(X)
        final_loss_sgd = loss_fn(pred_sgd, target)
        final_loss_adam = loss_fn(pred_adam, target)

    print("\n作业 SGD vs Adam")
    print("SGD final loss:", final_loss_sgd.item(), "w:", model_sgd.weight.item(), "b:", model_sgd.bias.item())
    print("Adam final loss:", final_loss_adam.item(), "w:", model_adam.weight.item(), "b:", model_adam.bias.item())
    # 对比：两者更新策略不一样，即使lr相同，收敛曲线、最终参数会不同。
    # Adam自适应步长，前期下降更快；SGD简单稳定，不一定谁更好。


if __name__ == "__main__":
    print("--- SGD 单步示例 ---")
    demo_one_sgd_step()

    print("\n--- 练习 1：一次参数更新 ---")
    practice_one_update()

    print("\n--- 练习 2：短训练循环 ---")
    practice_training_loop()

    print("\n--- 作业：SGD 与 Adam 对比 ---")
    homework_compare_optimizers()
