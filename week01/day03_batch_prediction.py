"""Week 1 / Day 3: batch prediction with plain Python."""

# 计算用的函数
def multi_linear(features, weights, bias):
    """Return w1*x1 + w2*x2 + ... + bias for one sample."""
    if len(features) != len(weights):
        raise ValueError("features 和 weights 的长度必须相同")

    total = 0
    for feature, weight in zip(features, weights):
        total += feature * weight

    return total + bias

# 储存结果
def batch_predict(samples, weights, bias):
    """Return one prediction for every sample in samples."""
    predictions = []

    for features in samples:
        prediction = multi_linear(features, weights, bias)
        predictions.append(prediction)

    return predictions


def demo_batch_prediction():
    """Predict three toy house-price samples as one batch."""
    samples = [
        [60, 1, 20],
        [80, 2, 10],
        [100, 3, 5],
    ]
    weights = [0.8, 15, -1.2]
    bias = 20

    predictions = batch_predict(samples, weights, bias)

    for index, prediction in enumerate(predictions, start=1):
        print("样本", index, "预测结果:", prediction)


def practice_batch_prediction():
    """练习 1：完成一批学生成绩的预测。"""
    # 每个样本依次表示：学习小时数、作业完成数、缺课次数。
    samples = [
        [2, 5, 0],
        [4, 4, 1],
        [1, 2, 3],
    ]
    weights = [5, 3, -4]
    bias = 40

    # TODO:
    # 1. 调用 batch_predict 得到 predictions。
    # 2. 用 enumerate(predictions, start=1) 依次打印每位学生的预测分数。
    predictions= batch_predict(samples, weights, bias)
    for index, prediction in enumerate(predictions, start=1):
        print(index, prediction)

def homework_custom_batch():
    """作业：自己设计一个两特征的批量预测问题。"""
    # TODO:
    # 1. 自己定义至少 3 个样本，每个样本只能有 2 个特征。
    # 2. 自己定义长度为 2 的 weights 和一个 bias。
    # 3. 调用 batch_predict。
    # 4. 打印 samples 和对应的 predictions。
    #
    # 可以使用“学习小时数和睡眠时间预测状态”或任意你感兴趣的例子。
    samples = [
        [4, 1],
        [2, 3],
        [6, 0],
    ]
    # 2. weights长度为2，bias
    weights = [6, -5]
    bias = 50

    # 3.调用批量预测
    predictions = batch_predict(samples, weights, bias)

    # 4.打印样本和预测结果
    for idx, (sample, pred) in enumerate(zip(samples, predictions), start=1):
        print(f"样本{idx} 特征{sample}，预测分数：{pred}")


if __name__ == "__main__":
    print("--- 批量预测示例 ---")
    demo_batch_prediction()

    print("\n--- 练习 1 ---")
    practice_batch_prediction()

    print("\n--- 作业 ---")
    homework_custom_batch()
