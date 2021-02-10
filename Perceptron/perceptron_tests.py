import perceptron as p1
from itertools import product

inputs_2 = [[0], [1]]

inputs_4 = [[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]]

inputs_8 = [list(combination) for combination in product([0, 1], repeat=3)]

# NOT-GATE
not_gate = p1.Perceptron("Not gate", [-1], 0)
print("Not-poort:")
for i in inputs_2:
    print(f"{i} --> {not_gate.calc(i)}")
print()

# AND-GATE
and_gate = p1.Perceptron("And gate", [0.5, 0.5], -1)
print("And-poort:")
for i in inputs_4:
    print(f"{i} --> {and_gate.calc(i)}")
print()

# OR-GATE
or_gate = p1.Perceptron("Or gate", [0.5, 0.5], -0.5)
print("Or-poort:")
for i in inputs_4:
    print(f"{i} --> {or_gate.calc(i)}")
print()

# NOR-GATE
nor_gate = p1.Perceptron("Nor gate", [-1, -1, -1], 0)
print("Nor-poort:")
for i in inputs_8:
    print(f"{i} --> {nor_gate.calc(i)}")
print()

# PARTY-GATE
# x1 = My friends are going to the party
# x2 = There are cats at the party
# x3 = There's no entry cost at the party
party_gate = p1.Perceptron("Party-gate 0 0 0", [0.6, 0.3, 0.2], -0.4)
print("Party-poort:")
for i in inputs_8:
    print(f"{i} --> {party_gate.calc(i)}")

