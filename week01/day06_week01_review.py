"""Week 1 / Day 6: integrated review project."""


class LinearRegressionModel:
    """A one-input linear regression model: y = weight * x + bias."""

    def __init__(self, weight, bias):
        # TODO: Save weight and bias as object attributes.
        self.weight = weight
        self.bias = bias

    def predict(self, x):
        """Return the prediction for one input."""
        # TODO: Return weight * x + bias using this model's parameters.
        return self.weight * x + self.bias

    def predict_batch(self, xs):
        """Return predictions for every x in xs."""
        # TODO:
        # 1. Create an empty predictions list.
        # 2. Loop through xs and call self.predict(x).
        # 3. Append each prediction and return the completed list.
        predictions = []
        for x in xs:
            predictions.append(self.predict(x))
        return predictions

    def mse(self, xs, ys):
        """Return mean squared error for the dataset."""
        # TODO:
        # 1. Check that xs and ys have the same length.
        # 2. Predict every x and calculate each squared error.
        # 3. Return the average squared error.
        if len(xs) != len(ys):
            raise ValueError("xs 和 ys 的长度必须相同")
        total_sq_err = 0
        for x,target in zip(xs,ys):
            pred=self.predict(x)
            total_sq_err += (pred-target)**2

        return total_sq_err / len(xs)

def find_best_parameters(xs, ys, candidate_weights, candidate_biases):
    """Return the candidate weight, bias, and loss with the lowest MSE."""
    # TODO:
    # 1. Set best_loss to float("inf").
    # 2. Set best_weight and best_bias to None.
    # 3. Try every weight/bias combination with two for loops.
    # 4. Create a LinearRegressionModel for each combination.
    # 5. Calculate its MSE and save it when it is better.
    # 6. Return best_weight, best_bias, best_loss.
    best_loss = float("inf")
    best_weight = None
    best_bias = None
    for w in candidate_weights:
        for b in candidate_biases:
            model=LinearRegressionModel(w, b)
            loss=model.mse(xs,ys)
            if loss < best_loss:
                best_loss = loss
                best_weight = w
                best_bias = b
    return best_weight, best_bias, best_loss




def main():
    """Run the Week 1 review project."""
    xs = [0, 1, 2, 3, 4]
    ys = [2, 5, 8, 11, 14]

    candidate_weights = [0, 1, 2, 3, 4, 5]
    candidate_biases = [0, 1, 2, 3, 4]

    # TODO:
    # 1. Call find_best_parameters(...).
    best_w, best_b, best_loss = find_best_parameters(xs, ys, candidate_weights, candidate_biases)
    # 2. Create the best model using the returned parameters.
    best_model = LinearRegressionModel(best_w, best_b)
    # 3. Print its predictions for xs.
    preds = best_model.predict_batch(xs)
    print("预测值：", preds)
    # 4. Print its best weight, bias, and MSE.
    print(f"best weight = {best_w}")
    print(f"best bias = {best_b}")
    print(f"best MSE = {best_loss}")
    # Expected results:
    # predictions = [2, 5, 8, 11, 14]
    # best weight = 3
    # best bias = 2
    # best MSE = 0.0


if __name__ == "__main__":
    main()
