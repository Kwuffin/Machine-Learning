def step(x):
    """
    Step function
    :param x: 64 bit integer
    :return:
    """
    return int(x >= 0)


class Perceptron:
    def __init__(self, name, weights, bias):
        self.name = name        # String
        self.weights = weights  # [float]
        self.bias = bias        # float
        self.output = 0

    # String function that prints useful attributes
    def __str__(self):
        print(f"Perceptron: '{self.name}':\n"
              f"Weight(s): {self.weights}\n"
              f"Bias:      {self.bias}")

    # Calculates its output
    def calc(self, inputs):
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
