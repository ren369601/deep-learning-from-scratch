"""
Week 1 / Day 1
Python linear model practice.

Goal:
Understand the simplest model form:

    y = w * x + b

This is the small idea behind a neural network Linear Layer.
"""


def linear(x, w, b):
    """Return y = w * x + b."""
    return w * x + b


def demo_single_value():
    w = 2
    b = 1
    x = 3

    y = linear(x, w, b)
    print("single value:")
    print(x, y)


def demo_list_values():
    xs = [1, 2, 3, 4, 5]

    print("list values:")
    for x in xs:
        y = linear(x, 2, 1)
        print(x, y)


def practice():
    # TODO:
    # Use xs = [1, 2, 3, 4, 5, 6]
    # Use y = 3 * x + 2
    # Print each x and y.
    xs=[1,2,3,4,5,6]
    for x in xs:
        y=linear(x, 3, 2)
        print(x, y)

def homework_function():
    # TODO:
    # Write a predict(x) function for y = 4 * x - 1.
    # Then use it on xs = [2, 4, 6, 8, 10].
    def predict(x):
        return 4*x-1
    xs=[2,4,6,8,10]
    for x in xs:
        y=predict(x)
        print(x, y)


class LinearModel:
    # TODO:
    # Complete this class.
    #
    # Example:
    # model = LinearModel(4, -1)
    # print(model.predict(2))
    def __init__(self, w, b):
        self.w=w
        self.b=b

    def predict(self, x):
        return self.w * x + self.b


if __name__ == "__main__":
    demo_single_value()
    demo_list_values()
    print("\n--- practice ---")
    practice()
    print("\n--- homework_function ---")
    homework_function()
    print("\n--- LinearModel class ---")
    model = LinearModel(4, -1)
    print(model.predict(2))

