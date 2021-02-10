from perceptron import Perceptron
from perceptronLayer import PerceptronLayer
from perceptronNetwork import PerceptronNetwork

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

# XOR-poort
xor_port = PerceptronNetwork([PerceptronLayer([Perceptron("1", [-0.5, -0.5], 0),  # Nor
                                               Perceptron("2", [0.5, 0.5], -1)]),  # And
                              PerceptronLayer([Perceptron("3", [-0.5, -0.5], 0)])])  # Nor
print("Xor-poort:")
for i in inputs:
    print(f"{i} --> {xor_port.feed_forward(i)}")
print()

# HALF ADDER
half_adder = PerceptronNetwork([PerceptronLayer([Perceptron("1", [-0.5, -0.5], 0),  # Nor
                                                 Perceptron("2", [0.5, 0.5], -1)]),  # And
                                PerceptronLayer([Perceptron("3", [0, 1], -1),
                                                 Perceptron("4", [-0.5, -0.5], 0)])])  # Nor
print("Half adder:")
print("        Carry, Sum")
for i in inputs:
    print(f"{i} --> {half_adder.feed_forward(i)}")
