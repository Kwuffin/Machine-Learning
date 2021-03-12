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
        for layer in self.layers:
            inputs = layer.update(inputs, learning_rate)

    def train(self, inputs: List[List[float]], targets: List[float], learning_rate: float):
        """
        Trains the neural network
        :param inputs: All training examples
        :param targets: Targets for every training example given
        :param learning_rate:
        :return: Amount of iterations
        """
        iterations = 0
        # Stop at 4000 iterations
        while iterations < 4000:
            iterations += 1
            outputs = []
            for input_index, training_example in enumerate(inputs):
                self.feed_forward(training_example)
                errors = []
                for i, layer in enumerate(reversed(self.layers)):
                    if i == 0:
                        for neuron in layer.neurons:
                            error = neuron.calc_error(training_example, targets[input_index])
                            errors.append(error)
                            outputs.append(neuron.output)

                    else:
                        for j, neuron in enumerate(layer.neurons):
                            error = neuron.calc_error_hidden(self.layers[i], j)
                            errors.append(error)
                            outputs.append(neuron.output)

                for e in errors:
                    if e < 0.00000001:
                        return iterations, errors, outputs
                self.update(training_example, learning_rate)

        return iterations, errors, outputs
