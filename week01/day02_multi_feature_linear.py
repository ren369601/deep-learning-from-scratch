"""Week 1 / Day 2: multi-feature linear model."""


def multi_linear(features, weights, bias):
    """Return w1*x1 + w2*x2 + ... + bias."""
    if len(features) != len(weights):
        raise ValueError("features 和 weights 的长度必须相同")

    total = 0
    for feature, weight in zip(features, weights):
        total += feature * weight

    return total + bias


def demo_house_price():
    """Use area, room count, and house age to make a toy prediction."""
    features = [80, 2, 10]
    weights = [0.8, 15, -1.2]
    bias = 20

    prediction = multi_linear(features, weights, bias)
    print("房屋特征:", features)
    print("预测结果:", prediction)


def demo_multiple_samples():
    """Predict values for several samples using the same model parameters."""
    samples = [
        [60, 1, 20],
        [80, 2, 10],
        [100, 3, 5],
    ]
    weights = [0.8, 15, -1.2]
    bias = 20

    for features in samples:
        prediction = multi_linear(features, weights, bias)
        print(features, "->", prediction)


def practice_student_score():
    """练习 1：完成一个三特征的分数预测。"""
    # 设定：features 依次是学习小时数、作业完成数、缺课次数。
    # 请自行设定三个 weights 和一个 bias，调用 multi_linear 并打印结果。
    # 提示：缺课次数对应的权重可以是负数。
    features=[10,8,2]
    weights=[5,3,-4]
    bias=40
    prediction = multi_linear(features, weights, bias)
    print(prediction)



class MultiFeatureLinearModel:
    """练习 2：把多输入线性模型封装成一个类。"""

    def __init__(self, weights, bias):
        # TODO: 把 weights 和 bias 保存为对象自己的属性。
        self.weights = weights
        self.bias = bias

    def predict(self, features):
        # TODO: 调用 multi_linear，返回预测结果。
        return multi_linear(features, self.weights, self.bias)


def homework_model_class():
    """作业：创建模型，预测两个不同样本。"""
    # TODO:
    # 1. 创建 MultiFeatureLinearModel，使用 weights = [2, 3]，bias = 1。
    # 2. 分别预测 [1, 2] 和 [3, 4]。
    # 3. 打印两个预测结果。
    model1=MultiFeatureLinearModel(weights=[2,3], bias=1)
    pred1=model1.predict([1,2])
    pred2=model1.predict([3,4])
    print(pred1)
    print(pred2)

if __name__ == "__main__":
    print("--- 房价示例 ---")
    demo_house_price()

    print("\n--- 多个样本 ---")
    demo_multiple_samples()

    print("\n--- 练习 1 ---")
    practice_student_score()

    print("\n--- 作业：模型类 ---")
    homework_model_class()
