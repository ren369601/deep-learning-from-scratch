"""Week 1 / Day 4: organize a linear model with a class."""


class LinearModel:
    """A plain-Python linear model for multi-feature inputs."""

    def __init__(self, weights, bias):
        # TODO: Save weights and bias as this model's attributes.
        self.weights=weights
        self.bias=bias

    def predict(self, features):
        """Return one prediction for one sample."""
        # TODO:
        # 1. Check whether len(features) equals len(self.weights).
        # 2. If not, raise ValueError("features 和 weights 的长度必须相同").
        # 3. Calculate and return feature * weight sums plus self.bias.
        if len(features) != len(self.weights):
            raise ValueError("features 和 weights 的长度必须相同")

        total=0
        for feature, weight in zip(features, self.weights):
            total += weight * feature

        return total+self.bias

    def predict_batch(self, samples):
        """Return one prediction for every sample."""
        # TODO:
        # 1. Create an empty predictions list.
        # 2. Loop through samples.
        # 3. Use self.predict(features) for each sample.
        # 4. Append every result and return predictions.
        predictions=[]
        for sample in samples:
            pred=self.predict(sample)
            predictions.append(pred)

        return predictions

    def set_parameters(self, weights, bias):
        """Replace this model's parameters."""
        # TODO: Update this model's weights and bias.
        self.weights=weights
        self.bias=bias

# 赋值，计算，预测结果，调整参数


def demo_expected_usage():
    """Read this function first; run it after you finish LinearModel."""
    model = LinearModel(weights=[5, 3, -4], bias=40)
    samples = [
        [2, 5, 0],
        [4, 4, 1],
        [1, 2, 3],
    ]

    print("模型参数:", model.weights, model.bias)
    print("单样本预测:", model.predict(samples[0]))
    print("批量预测:", model.predict_batch(samples))


def practice_parameter_update():
    """练习：观察更新参数前后的预测变化。"""
    # TODO:
    # 1. 创建 LinearModel(weights=[2, 3], bias=1)。
    # 2. 预测 [1, 2]，并打印“更新前”。
    # 3. 使用 set_parameters([4, 1], 0) 更新模型。
    # 4. 再预测 [1, 2]，并打印“更新后”。
    #
    # 正确的两个预测结果应分别为 9 和 6。
    model=LinearModel(weights=[2,3],bias=1)
    pred1=model.predict([1,2])
    print(pred1)

    model.set_parameters(weights=[4,1],bias=0)
    pred2=model.predict([1,2])
    print(pred2)

def homework_two_models():
    """作业：创建两个独立模型，并比较它们的预测。"""
    # TODO:
    # 1. 创建 model_a，weights=[1, 2]，bias=0。
    # 2. 创建 model_b，weights=[3, -1]，bias=5。
    # 3. 对同一个 features=[2, 4] 分别进行预测。
    # 4. 打印两个模型的预测结果。
    #
    # 思考：输入相同，为什么两个结果不同？
    model_a=LinearModel(weights=[1,2],bias=0)
    model_b=LinearModel(weights=[3,-1],bias=5)
    pred1=model_a.predict([2,4])
    pred2=model_b.predict([2,4])
    print(pred1)
    print(pred2)

if __name__ == "__main__":
    print("--- 示例：完成类后应得到的结果 ---")
    demo_expected_usage()

    print("\n--- 练习：更新参数 ---")
    practice_parameter_update()

    print("\n--- 作业：两个模型 ---")
    homework_two_models()
