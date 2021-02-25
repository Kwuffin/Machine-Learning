from math import e


def sigmoid(x):
    """
    Sigmoid function
    :param x: any integer value
    :return:
    """
    return 1/(1+e**(-x))


class Neuron:
    def __init__(self, name: str, weights: [float], bias: float):
        self.name = name
        self.weights = weights
        self.bias = bias

    def __str__(self) -> str:
        return f"Neuron '{self.name}':\nWeight(s): {self.weights}\nBias:      {self.bias}\n"

    def calc(self, inputs: [float]):
        """
        Calculates the neurons output
        :param inputs: Output values of the previous layer (or first set of inputs)
        :return: Output of neuron
        """
        # Error catching if length of lists is not the same
        if len(inputs) == len(self.weights):
            weighted = []  # List of all weighted inputs

            # Weigh all inputs
            for i in range(len(inputs)):
                weighted_input = inputs[i] * self.weights[i]
                weighted.append(weighted_input)

            total = sum(weighted) + self.bias  # Add bias to sum of weighted inputs

            return sigmoid(total)

        # If there are not enough weights for inputs, or vice versa.
        else:
            raise Exception(f"Inputs and Weights do not have an equal amount of elements\n"
                            f"Inputs:  {len(inputs)}\n"
                            f"Weights: {len(self.weights)}")
