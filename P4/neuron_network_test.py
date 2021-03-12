from neuronNetwork import NeuronNetwork
from neuronLayer import NeuronLayer
from neuron import Neuron
from random import randint

net1 = NeuronNetwork([NeuronLayer([Neuron("1", [0, 0], 0), Neuron("2", [-1, -1], -1)]),
                      NeuronLayer([Neuron("3", [1, 1], 1)])])


input_combinations_2 = [[0, 0], [1, 0], [0, 1], [1, 1]]
input_combinations_3 = [[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1]]


def and_port_train(inputs):
    targets = [0, 0, 0, 1]

    and_port = NeuronNetwork([
        NeuronLayer([
            Neuron("And-gate", [randint(-1, 1), randint(-1, 1)], randint(-1, 1))])])

    iterations, errors, outputs = and_port.train(inputs, targets, 1)

    print([neuron.weights for layer in and_port.layers for neuron in layer.neurons])
    print(iterations, errors, outputs)


def xor_port_train(inputs):
    targets = [0, 1, 1, 0]
    xor_port = NeuronNetwork([
        NeuronLayer([
            Neuron("Nor gate", [randint(-1, 1), randint(-1, 1)], randint(-1, 1)),
            Neuron("And gate", [randint(-1, 1), randint(-1, 1)], randint(-1, 1))
        ]),
        NeuronLayer([
            Neuron("Nor gate", [randint(-1, 1), randint(-1, 1)], randint(-1, 1))
        ])
    ])

    iterations, errors, outputs = xor_port.train(inputs, targets, 1)

    layer = xor_port.layers[-1]

    print([neuron.weights for neuron in layer.neurons])
    print(iterations, errors, outputs)


and_port_train(input_combinations_2)
xor_port_train(input_combinations_2)
