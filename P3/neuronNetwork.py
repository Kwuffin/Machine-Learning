from neuronLayer import NeuronLayer


class NeuronNetwork:
    def __init__(self, neuronLayers: [NeuronLayer]):
        self.layers = neuronLayers

    def __str__(self):
        return f"This network exists of {len(self.layers)} layers"

    def feed_forward(self, inputs):
        for layer in self.layers:
            inputs = layer.calc(inputs)

        return inputs
