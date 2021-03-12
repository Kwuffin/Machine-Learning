from neuron import Neuron
from typing import List


class NeuronLayer:
    def __init__(self, neurons: List[Neuron]):
        self.neurons = neurons

    def calc(self, inputs):
        return [neuron.calc(inputs) for neuron in self.neurons]

    def update(self, inputs: List[float], learning_rate: float):
        for neuron in self.neurons:
            neuron.update(inputs, learning_rate)
