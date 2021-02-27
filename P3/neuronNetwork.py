from neuronLayer import NeuronLayer


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
