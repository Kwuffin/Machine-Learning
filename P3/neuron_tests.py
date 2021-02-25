from neuron import Neuron
from itertools import product

inputs_2 = [[0], [1]]

inputs_4 = [[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]]

inputs_8 = [list(combination) for combination in product([0, 1], repeat=3)]


print("Neurons with the same values as Perceptrons:")
# NOT-GATE
not_gate = Neuron("Not gate", [-1], 0)
print("Not-poort:")
for i in inputs_2:
    print(f"{i} --> {not_gate.calc(i)}")
print()

# AND-GATE
and_gate = Neuron("And gate", [0.5, 0.5], -1)
print("And-poort:")
for i in inputs_4:
    print(f"{i} --> {and_gate.calc(i)}")
print()

# OR-GATE
or_gate = Neuron("Or gate", [0.5, 0.5], -0.5)
print("Or-poort:")
for i in inputs_4:
    print(f"{i} --> {or_gate.calc(i)}")
print()

######### UITLEG:
"""
De rede dat Neuronen niet werken met dezelfde waarden als de Perceptronen, is omdat neuronen een sigmoid-
functie gebruiken en kunnen dus ook waardes tussen de 0 en 1 teruggeven. We moeten een 'x' hebben die meer
op de extremen van de sigmoid-functie zitten, bijvoorbeeld x=20.
"""


print("Neurons with altered values:")
# NOT-GATE
not_gate = Neuron("Not gate", [-20], 10)
print("Not-poort:")
for i in inputs_2:
    print(f"{i} --> {round(not_gate.calc(i))} (actual value: {not_gate.calc(i)})")
print()

# AND-GATE
and_gate = Neuron("And gate", [75, 75], -100)
print("And-poort:")
for i in inputs_4:
    print(f"{i} --> {round(and_gate.calc(i))} (actual value: {and_gate.calc(i)})")
print()

# OR-GATE
or_gate = Neuron("Or gate", [75, 75], -100)
print("Or-poort:")
for i in inputs_4:
    print(f"{i} --> {round(or_gate.calc(i))} (actual value: {or_gate.calc(i)})")
print()

# NOR-GATE
nor_gate = Neuron("Nor gate", [-100, -100, -100], 50)
print("Nor-poort:")
for i in inputs_8:
    print(f"{i} --> {round(nor_gate.calc(i))} (actual value: {nor_gate.calc(i)})")
print()
