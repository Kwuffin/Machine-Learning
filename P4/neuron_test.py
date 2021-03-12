from neuron import Neuron

neuron1 = Neuron("1", [1, -1], 0)
learning_rate = 0.3
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [0, 0, 0, 1]

for i in range(len(inputs)):
    neuron1.calc_error(inputs[i], targets[i])
    print(neuron1.error)

