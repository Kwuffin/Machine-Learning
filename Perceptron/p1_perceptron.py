class Perceptron:
    #                  [int], str , [float] , float
    def __init__(self, inputs, name, weights, bias):
        self.inputs = inputs
        self.name = name
        self.weights = weights
        self.bias = bias
        self.output = 0

    # String function that prints useful attributes
    def __str__(self):
        print(f"Perceptron '{self.name}':\n"
              f"Input(s):  {self.inputs}\n"
              f"Weight(s): {self.weights}\n"
              f"Bias:      {self.bias}")

    # Calculates its output
    def calc(self):
        # Error catching if length of lists is not the same
        if len(self.inputs) == len(self.weights):
            weighted = []  # List of all weighted inputs

            # Weigh all inputs
            for i in range(len(self.inputs)):
                weighted_input = self.inputs[i] * self.weights[i]
                weighted.append(weighted_input)

            total = sum(weighted) + self.bias  # Add bias to sum of weighted inputs

            if total >= 0:
                self.output += 1

            return self.output

        # If there are not enough weights for inputs, or vice versa.
        else:
            print("Inputs and weights do not have an equal amount of values!")
            exit()
