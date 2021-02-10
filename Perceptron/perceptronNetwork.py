class PerceptronNetwork:
    def __init__(self, perceptronLayers):
        self.layers = perceptronLayers  # [PerceptronLayer]

    def feed_forward(self, inputs):
        for layer in self.layers:
            inputs = layer.calc(inputs)

        return inputs
