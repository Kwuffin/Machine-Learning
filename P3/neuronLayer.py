from neuron import Neuron


class NeuronLayer:
    def __init__(self, neurons: [Neuron]):
        self.neurons = neurons

    def calc(self, inputs):
        return [neuron.calc(inputs) for neuron in self.neurons]
