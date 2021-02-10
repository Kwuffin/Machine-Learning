class PerceptronLayer:
    def __init__(self, perceptrons):
        self.perceptrons = perceptrons  # [Perceptron]

    def calc(self, inputs):
        return [perceptron.calc(inputs) for perceptron in self.perceptrons]
