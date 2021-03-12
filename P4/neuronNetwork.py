from neuronLayer import NeuronLayer
from typing import List


class NeuronNetwork:
    def __init__(self, neuronLayers: [NeuronLayer]):
        self.layers = neuronLayers

    def __str__(self):
        return f"This network exists of {len(self.layers)} layers.\n" \
               f"It takes {len(self.layers[0])} inputs.\n" \
               f"It gives {len(self.layers[-1])} outputs."

    def feed_forward(self, inputs):
        for layer in self.layers:
            inputs = layer.calc(inputs)

        return inputs

    def update(self, inputs, learning_rate):
        """
        Updates all weights and biases
        :param inputs: The training example
        :param learning_rate:
        :return:
        """
        for layer in self.layers:
            layer.update(inputs, learning_rate)

    def train(self, inputs: List[List[float]], targets: List[float], learning_rate: float, user_iterations: int):
        """
        Trains the neural network
        :param user_iterations: Maximum amount of iterations
        :param inputs: All training examples
        :param targets: Targets for every training example given
        :param learning_rate:
        :return: Amount of iterations
        """
        iterations = 0
        # Stop at 4000 iterations
        while iterations < user_iterations:
            iterations += 1
            outputs = []  # Outputs of all training examples

            for input_index, training_example in enumerate(inputs):
                self.feed_forward(training_example)  # The feed forward makes sure all neurons have an error attribute
                errors = []  # All errors in a list

                for i, layer in enumerate(reversed(self.layers)):

                    # If current layer is the output layer
                    if i == 0:
                        # Calculate the output of every output neuron
                        for neuron_index, neuron in enumerate(layer.neurons):

                            # If the output layer has one output neuron
                            # print(type(targets[input_index]))
                            if isinstance(targets[input_index], list):
                                error = neuron.calc_error(training_example, targets[input_index][neuron_index])
                                errors.append(error)
                                outputs.append(neuron.output)

                            else:
                                error = neuron.calc_error(training_example, targets[input_index])
                                errors.append(error)
                                outputs.append(neuron.output)

                    else:
                        # Calculate the output of every hidden neuron
                        for j, neuron in enumerate(layer.neurons):
                            error = neuron.calc_error_hidden(self.layers[i], j)
                            errors.append(error)

                # Update weights and biases
                self.update(training_example, learning_rate)

        return iterations, errors, outputs
