import unittest
import p1_perceptron as p1


class TestLogicPorts(unittest.TestCase):

    def test_not(self):
        count = 1
        for i in range(2):
            not_on = p1.Perceptron([i], "Not gate", [-1], 0)
            self.assertEqual(count, not_on.calc())
            count -= 1

    def test_and(self):
        and_00 = p1.Perceptron([0, 0], "And gate 0 0", [0.5, 0.5], -1)
        and_01 = p1.Perceptron([0, 1], "And gate 0 1", [0.5, 0.5], -1)
        and_10 = p1.Perceptron([1, 0], "And gate 1 0", [0.5, 0.5], -1)
        and_11 = p1.Perceptron([1, 1], "And gate 1 1", [0.5, 0.5], -1)

        self.assertEqual(0, and_00.calc())
        self.assertEqual(0, and_01.calc())
        self.assertEqual(0, and_10.calc())
        self.assertEqual(1, and_11.calc())

    def test_or(self):
        or_00 = p1.Perceptron([0, 0], "Or gate 0 0", [0.5, 0.5], -0.5)
        or_01 = p1.Perceptron([0, 1], "Or gate 0 1", [0.5, 0.5], -0.5)
        or_10 = p1.Perceptron([1, 0], "Or gate 1 0", [0.5, 0.5], -0.5)
        or_11 = p1.Perceptron([1, 1], "Or gate 1 1", [0.5, 0.5], -0.5)

        self.assertEqual(0, or_00.calc())
        self.assertEqual(1, or_01.calc())
        self.assertEqual(1, or_10.calc())
        self.assertEqual(1, or_11.calc())

    def test_nor(self):
        nor_000 = p1.Perceptron([0, 0, 0], "NOR-gate 0 0 0", [-1, -1, -1], 0)
        nor_001 = p1.Perceptron([0, 0, 1], "NOR-gate 0 0 1", [-1, -1, -1], 0)
        nor_010 = p1.Perceptron([0, 1, 0], "NOR-gate 0 1 0", [-1, -1, -1], 0)
        nor_011 = p1.Perceptron([0, 1, 1], "NOR-gate 0 1 1", [-1, -1, -1], 0)
        nor_100 = p1.Perceptron([1, 0, 0], "NOR-gate 1 0 0", [-1, -1, -1], 0)
        nor_101 = p1.Perceptron([1, 0, 1], "NOR-gate 1 0 1", [-1, -1, -1], 0)
        nor_110 = p1.Perceptron([1, 1, 0], "NOR-gate 1 1 0", [-1, -1, -1], 0)
        nor_111 = p1.Perceptron([1, 1, 1], "NOR-gate 1 1 1", [-1, -1, -1], 0)

        # The only percepton that should give 1 as output is 000...
        self.assertEqual(1, nor_000.calc())

        # ...the rest should return 0
        off = [nor_001, nor_010, nor_011, nor_100, nor_101, nor_110, nor_111]
        for perceptron in off:
            self.assertEqual(0, perceptron.calc())

    def test_party(self):
        # x1 = My friends are going to the party
        # x2 = There are cats at the party
        # x3 = There's no entry cost at the party
        party_000 = p1.Perceptron([0, 0, 0], "Party-gate 0 0 0", [0.6, 0.3, 0.2], -0.4)
        party_001 = p1.Perceptron([0, 0, 1], "Party-gate 0 0 1", [0.6, 0.3, 0.2], -0.4)
        party_010 = p1.Perceptron([0, 1, 0], "Party-gate 0 1 0", [0.6, 0.3, 0.2], -0.4)
        party_011 = p1.Perceptron([0, 1, 1], "Party-gate 0 1 1", [0.6, 0.3, 0.2], -0.4)
        party_100 = p1.Perceptron([1, 0, 0], "Party-gate 1 0 0", [0.6, 0.3, 0.2], -0.4)
        party_101 = p1.Perceptron([1, 0, 1], "Party-gate 1 0 1", [0.6, 0.3, 0.2], -0.4)
        party_110 = p1.Perceptron([1, 1, 0], "Party-gate 1 1 0", [0.6, 0.3, 0.2], -0.4)
        party_111 = p1.Perceptron([1, 1, 1], "Party-gate 1 1 1", [0.6, 0.3, 0.2], -0.4)

        off = [party_000, party_001, party_010]
        on = [party_011, party_100, party_101, party_110, party_111]

        # All perceptrons that should output 0
        for perceptron in off:
            self.assertEqual(0, perceptron.calc())

        # All perceptrons that should output 1
        for perceptron in on:
            self.assertEqual(1, perceptron.calc())
