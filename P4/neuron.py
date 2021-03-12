from math import e
from typing import List


def sigmoid(x: float):
    """
    Sigmoid function
    :param x: any integer value
    :return:
    """
    return 1/(1+e**(-x))


class Neuron:
    def __init__(self, name: str, weights: List[float], bias: float):
        self.name = name
        self.weights = weights
        self.bias = bias
        self.error = None
        self.output = None

    def __str__(self) -> str:
        return f"Neuron '{self.name}':\nWeight(s): {self.weights}\nBias:      {self.bias}\n"

    def calc(self, inputs: List[float]) -> float:
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
            out = sigmoid(total)

            self.output = out
            return out

        # If there are not enough weights for inputs, or vice versa.
        else:
            raise Exception(f"Inputs and Weights do not have an equal amount of elements\n"
                            f"Inputs:  {len(inputs)}\n"
                            f"Weights: {len(self.weights)}")

    def calc_error(self, inputs: List[float], target: float) -> float:
        """
        Calculates the error for output neurons
        Δj = σ'(inputj) ∙ –(targetj – outputj)
        Δj = outputj ∙ (1 – outputj) ∙ –(targetj – outputj)
        :param inputs:
        :param target:
        :return: Error of neuron
        """
        activation = self.calc(inputs)

        sigmoid_dx = activation * (1 - activation)
        self.error = sigmoid_dx * -(target - activation)

        return self.error

    def calc_error_hidden(self, prev_neurons, neuron_count: int) -> float:
        """
        Calculates the error for hidden neurons
        Δi = σ'(inputi) ∙ Σj wi,j ∙ Δj
        :param prev_neurons:
        :param neuron_count:
        :return: Error of neuron
        """
        # Δi = σ'(inputi) ∙ Σj wi,j ∙ Δj
        # Δi = outputi ∙ (1 – outputi) ∙ Σj wi,j ∙ Δj
        weighted_errors = []
        for neuron in prev_neurons.neurons:
            weighted_error = neuron.error * neuron.weights[neuron_count]
            weighted_errors.append(weighted_error)
        sum_errors = sum(weighted_errors)

        sigmoid_dx = self.output * (1 - self.output)

        self.error = sigmoid_dx * sum_errors

        return self.error

    def update(self, inputs: List[float], learning_rate: float):
        """
        Calculates the new weights and biases, and applies them
        :param inputs:
        :param learning_rate:
        :return:
        """
        # Calculate deltas for weights
        # Δwi,j = η ∙ outputi ∙ Δj
        weight_deltas = []
        for i in inputs:
            weight_delta = learning_rate * i * self.error
            weight_deltas.append(weight_delta)

        # Apply new weights + weight delta's
        # w'i,j = wi,j – Δwi,j
        for i, weight in enumerate(self.weights):
            self.weights[i] -= weight_deltas[i]

        # Calculate and apply bias delta
        # Δbj = η ∙ Δj
        # b'j = bj – Δbj
        self.bias -= (learning_rate * self.error)
