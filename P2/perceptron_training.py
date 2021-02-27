from perceptron import Perceptron
from random import randint  # Is used to initialize perceptron classes with random values
from random import seed  # Is used to set the seed for the iris dataset
from sklearn.datasets import load_iris


def train_perceptron(gate, bit_combinations, targets):
    epochs = 0
    while True:
        values = []
        for i in range(len(bit_combinations)):
            a = gate.calc(bit_combinations[i])
            values.append(a)

            gate.update(target=targets[i], inputs=bit_combinations[i], learning_rate=0.3)

            epochs += 1

        if values == targets:
            print(f"Target outputs have been reached after {epochs} iteration(s).\n")
            print(f"Final weights and bias:\n"
                  f"{gate.__str__()}")
            break

        if epochs == 1000:
            print(f"No correct weights and biases found within 4000 iterations for '{gate.name}'.")
            break


def and_gate_train():
    bit_combinations = [[1, 1], [1, 0], [0, 1], [0, 0]]

    and_targets = [0, 0, 0, 1]
    and_gate = Perceptron("And gate", [randint(-5, 5), randint(-5, 5)], randint(-5, 5))

    train_perceptron(and_gate, bit_combinations, and_targets)


def xor_gate_train():
    bit_combinations = [[1, 1], [1, 0], [0, 1], [0, 0]]

    and_targets = [0, 1, 1, 0]
    xor_gate = Perceptron("Xor gate", [randint(-5, 5), randint(-5, 5)], randint(-5, 5))

    train_perceptron(xor_gate, bit_combinations, and_targets)


def iris_train():
    seed(1763456)
    iris_perceptron = Perceptron("Iris", [randint(-5, 5), randint(-5, 5), randint(-5, 5), randint(-5, 5)], randint(-5, 5))

    data = load_iris()
    inputs = list(data.data[:100])
    targets = list(data.target[:100])

    train_perceptron(iris_perceptron, inputs, targets)



def main():
    # print("====================| And-Gate: |====================")
    # and_gate_train()
    # print("====================| Xor-Gate: |====================")
    # xor_gate_train()
    iris_train()


if __name__ == '__main__':
    main()
