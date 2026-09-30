"""Week 1 / Day 5: a plain-Python linear-regression prototype."""


def linear(x, weight, bias):
    """Return one linear prediction: weight * x + bias."""
    return weight * x + bias


def mean_squared_error(xs, ys, weight, bias):
    """Calculate MSE for a dataset using one pair of parameters."""
    if len(xs) != len(ys):
        raise ValueError("xs 和 ys 的长度必须相同")

    total_squared_error = 0

    for x, target in zip(xs, ys):
        prediction = linear(x, weight, bias)
        squared_error = (prediction - target) ** 2
        total_squared_error += squared_error

    return total_squared_error / len(xs)


def demo_compare_two_models():
    """Compare one poor model with one perfect model."""
    xs = [1, 2, 3, 4]
    ys = [3, 5, 7, 9]

    poor_loss = mean_squared_error(xs, ys, weight=1, bias=0)
    perfect_loss = mean_squared_error(xs, ys, weight=2, bias=1)

    print("w=1, b=0 的 MSE:", poor_loss)
    print("w=2, b=1 的 MSE:", perfect_loss)


def practice_compare_models():
    """练习：自己比较两组参数的 MSE。"""
    xs = [1, 2, 3, 4]
    ys = [3, 5, 7, 9]

    # TODO:
    # 1. 计算 weight=3, bias=0 的 MSE。
    # 2. 计算 weight=2, bias=2 的 MSE。
    # 3. 打印两个 MSE，并说出哪一个模型更好。
    #
    # 提示：MSE 更小的模型更好。
    loss1= mean_squared_error(xs, ys, weight=3, bias=0)
    loss2= mean_squared_error(xs, ys, weight=2, bias=2)
    print(loss1)
    print(loss2)
    if loss1 > loss2:
        print("第二个模型更好")
    else:
        print("第一个模型更好")


def homework_find_best_model():
    """作业：枚举候选参数，找到 MSE 最小的线性模型。"""
    xs = [1, 2, 3, 4]
    ys = [3, 5, 7, 9]
    candidate_weights = [0, 1, 2, 3, 4]
    candidate_biases = [-1, 0, 1, 2, 3]

    # TODO:
    # 1. 设置 best_loss = float("inf")。
    # 2. 设置 best_weight 和 best_bias，初始值可以是 None。
    # 3. 用两层 for 循环遍历所有 weight / bias 组合。
    # 4. 对每组参数计算 loss。
    # 5. 如果 loss < best_loss，更新 best_loss、best_weight、best_bias。
    # 6. 最后打印最优参数和最优 MSE。
    #
    # 预期最优结果：weight=2, bias=1, MSE=0.0。
    best_loss = float("inf")
    best_weight = None
    best_bias = None
    # 3. 两层for循环遍历所有 weight / bias 组合
    for w in candidate_weights:
        for b in candidate_biases:
            loss = mean_squared_error(xs, ys, w, b)
            # 5. 如果 loss < best_loss，更新
            if loss < best_loss:
                best_loss = loss
                best_weight = w
                best_bias = b
    # 6. 最后打印最优参数和最优 MSE。
    print(f"最优 weight={best_weight}, bias={best_bias}, MSE={best_loss}")


if __name__ == "__main__":
    print("--- 比较两个模型 ---")
    demo_compare_two_models()

    print("\n--- 练习：比较 MSE ---")
    practice_compare_models()

    print("\n--- 作业：寻找最优模型 ---")
    homework_find_best_model()
