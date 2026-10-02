"""Week 2 / Day 6: multi-feature linear regression with NumPy."""

import numpy as np


class NumpyLinearRegression:
    """A multi-feature linear regression model trained with gradient descent."""

    def __init__(self, n_features):
        # TODO:
        # 1. Create self.weights as float32 zeros with shape (n_features,).
        # 2. Set self.bias to 0.0.
        # 3. Create self.loss_history as an empty list.
        self.weights=np.zeros(n_features,dtype=np.float32)
        self.bias=0.0
        self.loss_history=[]

    def predict(self, X):
        """Return predictions for a feature matrix X."""
        # TODO:
        # 1. Check that X is two-dimensional.
        # 2. Check that X.shape[1] matches the number of weights.
        # 3. Return X @ self.weights + self.bias.
        assert X.ndim == 2, "X must be 2D matrix"
        assert X.shape[1] == self.weights.shape[0], "feature count mismatch"
        return X @ self.weights + self.bias

    @staticmethod
    def compute_loss(predictions, targets):
        """Return MSE for arrays with matching shapes."""
        # TODO:
        # 1. Check that the two shapes match.
        # 2. Return MSE as a Python float.
        assert predictions.shape == targets.shape, "shape mismatch"
        mse = np.mean((predictions - targets) ** 2)
        return float(mse)

    def compute_gradients(self, X, targets):
        """Return gradients for all weights and the bias."""
        # TODO:
        # 1. Calculate predictions and errors.
        # 2. Get n_samples from X.shape[0].
        # 3. Calculate grad_weights = (2 / n_samples) * (X.T @ errors).
        # 4. Calculate grad_bias = 2 * errors.mean().
        # 5. Return grad_weights and grad_bias.
        preds = self.predict(X)
        errors = preds - targets
        n_samples = X.shape[0]
        grad_weights = (2 / n_samples) * (X.T @ errors)
        grad_bias = 2 * errors.mean()
        return grad_weights, grad_bias

    def fit(self, X, targets, learning_rate, epochs, log_every=None):
        """Train the model and return its loss history."""
        # TODO:
        # 1. Clear self.loss_history before training.
        # 2. Repeat for the requested number of epochs:
        #    a. Predict and calculate loss.
        #    b. Append loss to self.loss_history.
        #    c. Calculate both gradients.
        #    d. Update self.weights and self.bias.
        #    e. If log_every is set, print progress at that interval.
        # 3. Return self.loss_history.
        self.loss_history.clear()
        for epoch in range(epochs):
            preds = self.predict(X)
            loss = self.compute_loss(preds, targets)
            self.loss_history.append(loss)

            grad_w, grad_b = self.compute_gradients(X, targets)
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b

            if log_every is not None and epoch % log_every == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
        return self.loss_history


def demo_matrix_gradients():
    """Show the shape of every value in a multi-feature gradient step."""
    X = np.array([
        [1, 2],
        [2, 0],
        [0, 3],
    ], dtype=np.float32)
    targets = np.array([3, 7, 0], dtype=np.float32)
    weights = np.zeros(2, dtype=np.float32)
    bias = 0.0

    predictions = X @ weights + bias
    errors = predictions - targets
    grad_weights = (2 / X.shape[0]) * (X.T @ errors)
    grad_bias = 2 * errors.mean()

    print("X:", X.shape)
    print("weights:", weights.shape)
    print("predictions:", predictions.shape)
    print("errors:", errors.shape)
    print("X.T:", X.T.shape)
    print("grad_weights:", grad_weights, grad_weights.shape)
    print("grad_bias:", grad_bias)


def practice_train_two_features():
    """练习：训练模型学习 y = 2*x1 - 3*x2 + 5。"""
    X = np.array([
        [-2, -1],
        [-1, 0],
        [0, 1],
        [1, 2],
        [2, -2],
        [-1, 2],
        [2, 1],
        [0, -2],
    ], dtype=np.float32)
    targets = X @ np.array([2, -3], dtype=np.float32) + 5

    # TODO:
    # 1. Create NumpyLinearRegression(n_features=2).
    # 2. Train with learning_rate=0.05 and epochs=1000.
    # 3. Print the first loss and final loss.
    # 4. Print learned weights and bias.
    # 5. Print predictions and targets.
    #
    # Expected: weights near [2, -3], bias near 5, final loss near 0.
    model = NumpyLinearRegression(n_features=2)
    model.fit(X, targets,0.05,1000)
    print(model.loss_history[0])
    print(model.loss_history[-1])
    print(model.weights)
    print(model.bias)
    preds=model.predict(X)
    print(preds)
    print(targets)



def homework_three_features():
    """作业：训练一个有三个输入特征的模型。"""
    X = np.array([
        [-2, -1, 0],
        [-1, 0, 1],
        [0, 1, 2],
        [1, 2, -1],
        [2, -2, 1],
        [-1, 2, 2],
        [2, 1, -2],
        [0, -2, -1],
    ], dtype=np.float32)
    true_weights = np.array([1.5, -2.0, 0.5], dtype=np.float32)
    true_bias = 4.0
    targets = X @ true_weights + true_bias

    # TODO:
    # 1. Before training, write down all important expected shapes.
    # 2. Create a model with n_features=3.
    # 3. Choose a learning rate and number of epochs, then train it.
    # 4. Print first/final loss, learned parameters, and predictions.
    # 5. Compare learned parameters with true_weights and true_bias.
    # 6. Confirm that loss decreases during training.
    print("\n=== Shape check ===")
    print(f"X shape {X.shape} (samples, 3 features)")
    print(f"targets shape {targets.shape}")

    model = NumpyLinearRegression(n_features=3)
    lr = 0.03
    epochs = 2000
    model.fit(X, targets, learning_rate=lr, epochs=epochs)

    first_loss = model.loss_history[0]
    final_loss = model.loss_history[-1]
    preds = model.predict(X)

    print(f"First loss: {first_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"True weights:   {true_weights}")
    print(f"Learned weights:{model.weights}")
    print(f"True bias:      {true_bias}")
    print(f"Learned bias:   {model.bias}")
    print(f"Predictions: {preds}")
    print(f"Targets:     {targets}")



if __name__ == "__main__":
    print("--- 矩阵梯度 shape 示例 ---")
    demo_matrix_gradients()

    print("\n--- 练习：两个特征 ---")
    practice_train_two_features()

    print("\n--- 作业：三个特征 ---")
    homework_three_features()
