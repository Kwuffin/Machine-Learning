from math import e


def step(x):
    """
    Step function
    :param x: any integer value
    :return:
    """
    return int(x >= 0)


def sigmoid(x):
    """
    Sigmoid function
    :param x: any integer value
    :return:
    """
    return 1/(1+e**(-x))


class Perceptron:
    def __init__(self, name: str, weights: [float], bias: float):
        self.name = name
        self.weights = weights
        self.bias = bias
        self.output = 0

    # String function that prints useful attributes
    def __str__(self):
        return f"Perceptron: '{self.name}':\nWeight(s): {self.weights}\nBias:      {self.bias}"

    # Calculates its output
    def calc(self, inputs: [int]) -> int:
        """
        Calculates the perceptron's output
        :param inputs: Output values of the previous layer (or first set of inputs)
        :return: Output of perceptron
        """
        # Error catching if length of lists is not the same
        if len(inputs) == len(self.weights):
            weighted = []  # List of all weighted inputs

            # Weigh all inputs
            for i in range(len(inputs)):
                weighted_input = inputs[i] * self.weights[i]
                weighted.append(weighted_input)

            total = sum(weighted) + self.bias  # Add bias to sum of weighted inputs

            return step(total)

        # If there are not enough weights for inputs, or vice versa.
        else:
            raise Exception(f"Inputs and Weights do not have an equal amount of elements\n"
                            f"Inputs:  {len(inputs)}\n"
                            f"Weights: {len(self.weights)}")

    def update(self, target: int, inputs: [int], learning_rate: float):
        """
        Perceptron Learning Rule, adjusts weights and bias
        :param target: The correct answer
        :param inputs: List
        :param learning_rate:
        """
        a = self.calc(inputs)

        if a != target:
            self.bias += learning_rate * (target - a) * 1

            new_weights = []
            for i in range(len(self.weights)):
                new_weight = self.weights[i] + (learning_rate * (target - a) * inputs[i])

            self.weights = new_weights
