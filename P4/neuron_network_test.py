from neuronNetwork import NeuronNetwork
from neuronLayer import NeuronLayer
from neuron import Neuron
from random import randint

input_combinations_2 = [[0, 0], [1, 0], [0, 1], [1, 1]]
input_combinations_3 = [[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1]]


def and_port_train(inputs, user_iteration, learning_rate):
    targets = [0, 0, 0, 1]

    and_port = NeuronNetwork([
        NeuronLayer([
            Neuron("And-gate", [randint(-1, 1), randint(-1, 1)], randint(-1, 1))])])

    iterations, errors, outputs = and_port.train(inputs, targets, learning_rate, user_iteration)

    print(f"============ | And gate | ============\n"
          f"After {iterations} iterations:\n"
          f"Errors: {errors}\n"
          f"Inputs: {inputs}\n"
          f"Outputs: {outputs}\n"
          f"Targets: {targets}\n"
          f"Weights:", end=' ')
    layer = and_port.layers[-1]
    print(f"{[neuron.weights for neuron in layer.neurons]}")
    print(f"Bias: {[neuron.bias for neuron in layer.neurons]}\n\n")


def xor_port_train(inputs, user_iteration, learning_rate):
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

    iterations, errors, outputs = xor_port.train(inputs, targets, learning_rate, user_iteration)

    print(f"============ | Xor gate | ============\n"
          f"After {iterations} iterations:\n"
          f"Errors: {errors}\n"
          f"Inputs: {inputs}\n"
          f"Outputs: {outputs}\n"
          f"Targets: {targets}\n"
          f"Weights:", end=' ')
    layer = xor_port.layers[-1]
    print([neuron.weights for neuron in layer.neurons])
    print(f"Bias: {[neuron.bias for neuron in layer.neurons]}\n\n")


def half_adder_train(inputs, user_iteration, learning_rate):
    targets = [[0, 0], [0, 1], [0, 1], [1, 0]]
    half_adder = NeuronNetwork([
        NeuronLayer([Neuron("1", [randint(-1, 1), randint(-1, 1)], randint(-1, 1)),
                     Neuron("2", [randint(-1, 1), randint(-1, 1)], randint(-1, 1))]),

        NeuronLayer([Neuron("3", [randint(-1, 1), randint(-1, 1)], randint(-1, 1)),
                     Neuron("4", [randint(-1, 1), randint(-1, 1)], randint(-1, 1))])])

    iterations, errors, outputs = half_adder.train(inputs, targets, learning_rate, user_iteration)

    print(f"============ | Half adder | ============\n"
          f"After {iterations} iterations:\n"
          f"Errors: {errors}\n"
          f"Inputs: {inputs}\n"
          f"Outputs: {outputs}\n"
          f"Targets: {targets}\n"
          f"Weights:", end=' ')
    layer = half_adder.layers[-1]
    print([neuron.weights for neuron in layer.neurons])
    print(f"Bias: {[neuron.bias for neuron in layer.neurons]}\n\n")


def main():
    # Iterations
    user_iterations = input("How many iterations?\n"
                            "(Leave empty for default=4000)\n> ")
    if user_iterations == "":
        user_iterations = 4000
    else:
        user_iterations = int(
            user_iterations)  # Ik heb geen try-except voor als er geen getallen ingevuld, wees niet stom pls ;-;

    # Learning rate
    learning_rate = input("Fill a learning rate:\n"
                          "(Leave empty for default=1)\n> ")
    if learning_rate == "":
        learning_rate = 1
    else:
        if "." in learning_rate:
            learning_rate = float(learning_rate)
        else:
            learning_rate = int(learning_rate)

    and_port_train(input_combinations_2, user_iterations, learning_rate)
    xor_port_train(input_combinations_2, user_iterations, learning_rate)
    half_adder_train(input_combinations_2, user_iterations, learning_rate)


if __name__ == '__main__':
    main()
