from neuronNetwork import NeuronNetwork
from neuronLayer import NeuronLayer
from neuron import Neuron
from random import randint

net1 = NeuronNetwork([NeuronLayer([Neuron("1", [0, 0], 0), Neuron("2", [-1, -1], -1)]),
                      NeuronLayer([Neuron("3", [1, 1], 1)])])

and_port = NeuronNetwork([
    NeuronLayer([
        Neuron("And-gate", [-0.5, 0.5], 1.5)])])

input_combinations = [[0, 0], [1, 0], [0, 1], [1, 1]]
targets = [0, 0, 0, 1]


def and_port_train(inputs, targets):
    iterations, errors, outputs = and_port.train(inputs, targets, 1)
    print([neuron.weights for layer in and_port.layers for neuron in layer.neurons])
    print(iterations, errors, outputs)


and_port_train(input_combinations, targets)
