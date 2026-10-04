"""Week 3 / Day 4: scalar backpropagation and gradient checking."""

import numpy as np


def relu(x):
    """Return scalar ReLU output."""
    return max(0.0, x)


def relu_derivative(x):
    """Return the chosen ReLU derivative: 1 above zero, otherwise 0."""
    return 1.0 if x > 0 else 0.0


def scalar_linear_loss(weight, x, bias, target):
    """Return squared error for one scalar linear model."""
    prediction = weight * x + bias
    return (prediction - target) ** 2


def numerical_gradient(function, value, epsilon=1e-5):
    """Estimate a scalar derivative with the centered-difference method."""
    loss_plus = function(value + epsilon)
    loss_minus = function(value - epsilon)
    return (loss_plus - loss_minus) / (2 * epsilon)


def demo_linear_backward():
    """Show a complete forward and backward pass for y = wx + b."""
    x = 2.0
    weight = 3.0
    bias = 1.0
    target = 10.0

    prediction = weight * x + bias
    loss = (prediction - target) ** 2

    grad_prediction = 2 * (prediction - target)
    grad_weight = grad_prediction * x
    grad_bias = grad_prediction

    print("prediction:", prediction)
    print("loss:", loss)
    print("grad_prediction:", grad_prediction)
    print("grad_weight:", grad_weight)
    print("grad_bias:", grad_bias)


def practice_single_weight():
    """练习 1：手算并更新只有一个 weight 的模型。"""
    x = 3.0
    target = 12.0
    weight = 2.0
    learning_rate = 0.05

    # TODO:
    # 1. Forward: prediction = weight*x，loss = (prediction-target)^2。
    # 2. Backward: grad_prediction 和 grad_weight。
    # 3. 更新一次 weight。
    # 4. 计算更新后的 prediction 和 loss。
    # 5. 打印全部结果，并确认新 loss 更小。

    # 1. Forward
    prediction = weight * x
    loss = (prediction - target) ** 2
    # 2. Backward
    grad_prediction = 2 * (prediction - target)
    grad_weight = grad_prediction * x
    # 3. 更新一次 weight
    weight = weight - learning_rate * grad_weight
    # 4. 更新后前向
    new_prediction = weight * x
    new_loss = (new_prediction - target) ** 2
    # 5.打印
    print(f"prediction: {prediction}, loss: {loss}")
    print(f"grad_prediction: {grad_prediction}, grad_weight: {grad_weight}")
    print(f"updated weight: {weight}, new_prediction: {new_prediction}, new_loss: {new_loss}")



def practice_relu_backward():
    """练习 2：观察 ReLU 如何传递或截断梯度。"""
    values = np.array([-2.0, 0.0, 3.0])
    upstream_gradients = np.array([5.0, 5.0, 5.0])

    # TODO:
    # 1. 对每个 value 调用 relu_derivative()，组成 derivatives array。
    # 2. grad_inputs = upstream_gradients * derivatives。
    # 3. 打印 values、derivatives 和 grad_inputs。
    # 4. 解释负数位置的梯度为什么是 0。

    # 1.求导数组
    derivatives = np.array([relu_derivative(v) for v in values])
    # 2.梯度相乘
    grad_inputs = upstream_gradients * derivatives
    # 3.打印
    print("values:", values)
    print("derivatives:", derivatives)
    print("grad_inputs:", grad_inputs)
    # 4.解释：负数位置relu导数为0，梯度被截断
    print("负数位置的ReLU导数=0，上游梯度乘0，梯度直接断掉。")


def practice_gradient_check():
    """练习 3：比较解析梯度和数值梯度。"""
    x = 2.0
    weight = 3.0
    bias = 1.0
    target = 10.0

    # TODO:
    # 1. 使用公式计算 analytical_grad_weight。
    # 2. 定义只接收 weight 的 loss function，可使用 lambda。
    # 3. 调用 numerical_gradient() 得到 numerical_grad_weight。
    # 4. 打印两个梯度。
    # 5. 使用 np.isclose(..., atol=1e-4) 检查是否接近。

    # 1.解析梯度 analytical_grad_weight
    pred = weight * x + bias
    grad_pred = 2 * (pred - target)
    analytical_grad_weight = grad_pred * x
    # 2.lambda 定义loss函数，只接收weight
    loss_fn = lambda w: scalar_linear_loss(w, x, bias, target)
    # 3.数值梯度
    numerical_grad_weight = numerical_gradient(loss_fn, weight)
    # 4.打印
    print("analytical_grad_weight:", analytical_grad_weight)
    print("numerical_grad_weight:", numerical_grad_weight)
    # 5.对比
    close = np.isclose(analytical_grad_weight, numerical_grad_weight, atol=1e-4)
    print("gradients are close?", close)


def homework_two_layer_backprop():
    """作业：手写一个标量两层网络的一次 Forward 和 Backward。"""
    x = 2.0
    target = 1.0
    w1 = 1.0
    b1 = -1.0
    w2 = 2.0
    b2 = 0.0
    learning_rate = 0.01

    # TODO - Forward:
    # 1. z1 = w1*x + b1。
    z1 = w1 * x + b1
    # 2. a1 = relu(z1)。
    a1 = relu(z1)
    # 3. prediction = w2*a1 + b2。
    prediction = w2 * a1 + b2
    # 4. loss = (prediction-target)^2。
    loss = (prediction - target) ** 2

    # TODO - Backward，必须按反向顺序：
    # 5. grad_prediction = 2*(prediction-target)。
    grad_prediction = 2 * (prediction - target)
    # 6. grad_w2、grad_b2、grad_a1。
    grad_w2 = grad_prediction * a1
    grad_b2 = grad_prediction
    grad_a1 = grad_prediction * w2
    # 7. grad_z1 = grad_a1 * relu_derivative(z1)。
    grad_z1 = grad_a1 * relu_derivative(z1)
    # 8. grad_w1、grad_b1。
    grad_w1 = grad_z1 * x
    grad_b1 = grad_z1

    # TODO - Update:
    # 9. 同时更新 w1、b1、w2、b2。
    w1_new = w1 - learning_rate * grad_w1
    b1_new = b1 - learning_rate * grad_b1
    w2_new = w2 - learning_rate * grad_w2
    b2_new = b2 - learning_rate * grad_b2

    # 10. 使用新参数重新 Forward 并计算 new_loss。
    z1_new = w1_new * x + b1_new
    a1_new = relu(z1_new)
    pred_new = w2_new * a1_new + b2_new
    new_loss = (pred_new - target) ** 2

    # 11. 打印所有中间值、梯度、新参数和两个 loss。
    print("\n==== Forward ====")
    print(f"z1={z1}, a1={a1}, prediction={prediction}, loss={loss}")
    print("\n==== Backward Gradients ====")
    print(f"grad_pred={grad_prediction}, grad_w2={grad_w2}, grad_b2={grad_b2}, grad_a1={grad_a1}")
    print(f"grad_z1={grad_z1}, grad_w1={grad_w1}, grad_b1={grad_b1}")
    print("\n==== Updated Params ====")
    print(f"w1_new={w1_new}, b1_new={b1_new}, w2_new={w2_new}, b2_new={b2_new}")
    print(f"old loss={loss}, new loss={new_loss}")
    # 12. 确认 new_loss 小于原 loss。
    print("new_loss < old_loss ?", new_loss < loss)





if __name__ == "__main__":
    print("--- 线性模型 Backward 示例 ---")
    demo_linear_backward()

    print("\n--- 练习 1：单参数更新 ---")
    practice_single_weight()

    print("\n--- 练习 2：ReLU Backward ---")
    practice_relu_backward()

    print("\n--- 练习 3：梯度检查 ---")
    practice_gradient_check()

    print("\n--- 作业：两层网络 Backprop ---")
    homework_two_layer_backprop()
