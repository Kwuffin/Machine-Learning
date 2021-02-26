from neuron import Neuron
from neuronLayer import NeuronLayer
from neuronNetwork import NeuronNetwork

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

# HALF ADDER
half_adder = NeuronNetwork([NeuronLayer([Neuron("1", [-100, -100], 50),  # Nor
                                         Neuron("2", [75, 75], -100)]),  # And
                            NeuronLayer([Neuron("3", [0, 150], -100),
                                         Neuron("4", [-100, -100], 50)])])  # Nor
print("Half adder with rounded values:")
print("        Carry, Sum")
for i in inputs:
    print(f"{i} --> {[round(out) for out in half_adder.feed_forward(i)]}")


print("\n\nHalf adder without rounded values:")
print("        Carry, Sum")
for i in inputs:
    print(f"{i} --> {half_adder.feed_forward(i)}")
