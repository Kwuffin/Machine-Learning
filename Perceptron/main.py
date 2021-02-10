from perceptron import Perceptron
from perceptronLayer import PerceptronLayer
from perceptronNetwork import PerceptronNetwork

network = PerceptronNetwork([PerceptronLayer([Perceptron("1", [-0.5, -0.5], 0),     # Nor
                                              Perceptron("2", [0.5, 0.5], -1)]),    # And
                             PerceptronLayer([Perceptron("3", [-0.5, -0.5], 0)])])  # Nor

print(network.feed_forward([1, 0]))