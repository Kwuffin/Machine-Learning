import unittest
import p1_perceptron as p1


class TestLogicPorts(unittest.TestCase):

    def test_not(self):
        count = 1
        for i in range(2):
            not_on = p1.Perceptron("Not gate", [-1], 0)
            self.assertEqual(count, not_on.calc([i]))
            count -= 1

    def test_and(self):
        and_00 = p1.Perceptron("And gate 0 0", [0.5, 0.5], -1)
        and_01 = p1.Perceptron("And gate 0 1", [0.5, 0.5], -1)
        and_10 = p1.Perceptron("And gate 1 0", [0.5, 0.5], -1)
        and_11 = p1.Perceptron("And gate 1 1", [0.5, 0.5], -1)

        self.assertEqual(0, and_00.calc([0, 0]))
        self.assertEqual(0, and_01.calc([0, 1]))
        self.assertEqual(0, and_10.calc([1, 0]))
        self.assertEqual(1, and_11.calc([1, 1]))

    def test_or(self):
        or_00 = p1.Perceptron("Or gate 0 0", [0.5, 0.5], -0.5)
        or_01 = p1.Perceptron("Or gate 0 1", [0.5, 0.5], -0.5)
        or_10 = p1.Perceptron("Or gate 1 0", [0.5, 0.5], -0.5)
        or_11 = p1.Perceptron("Or gate 1 1", [0.5, 0.5], -0.5)

        self.assertEqual(0, or_00.calc([0, 0]))
        self.assertEqual(1, or_01.calc([0, 1]))
        self.assertEqual(1, or_10.calc([1, 0]))
        self.assertEqual(1, or_11.calc([1, 1]))

    def test_nor(self):
        nor_000 = p1.Perceptron("NOR-gate 0 0 0", [-1, -1, -1], 0)
        nor_001 = p1.Perceptron("NOR-gate 0 0 1", [-1, -1, -1], 0)
        nor_010 = p1.Perceptron("NOR-gate 0 1 0", [-1, -1, -1], 0)
        nor_011 = p1.Perceptron("NOR-gate 0 1 1", [-1, -1, -1], 0)
        nor_100 = p1.Perceptron("NOR-gate 1 0 0", [-1, -1, -1], 0)
        nor_101 = p1.Perceptron("NOR-gate 1 0 1", [-1, -1, -1], 0)
        nor_110 = p1.Perceptron("NOR-gate 1 1 0", [-1, -1, -1], 0)
        nor_111 = p1.Perceptron("NOR-gate 1 1 1", [-1, -1, -1], 0)

        # The only percepton that should give 1 as output is 000...
        self.assertEqual(1, nor_000.calc([0, 0, 0]))

        # ...the rest should return 0
        off = [nor_001, nor_010, nor_011, nor_100, nor_101, nor_110, nor_111]
        off_inputs = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        for i in range(len(off)):
            self.assertEqual(0, off[i].calc(off_inputs[i]))

    def test_party(self):
        # x1 = My friends are going to the party
        # x2 = There are cats at the party
        # x3 = There's no entry cost at the party
        party_000 = p1.Perceptron("Party-gate 0 0 0", [0.6, 0.3, 0.2], -0.4)
        party_001 = p1.Perceptron("Party-gate 0 0 1", [0.6, 0.3, 0.2], -0.4)
        party_010 = p1.Perceptron("Party-gate 0 1 0", [0.6, 0.3, 0.2], -0.4)
        party_011 = p1.Perceptron("Party-gate 0 1 1", [0.6, 0.3, 0.2], -0.4)
        party_100 = p1.Perceptron("Party-gate 1 0 0", [0.6, 0.3, 0.2], -0.4)
        party_101 = p1.Perceptron("Party-gate 1 0 1", [0.6, 0.3, 0.2], -0.4)
        party_110 = p1.Perceptron("Party-gate 1 1 0", [0.6, 0.3, 0.2], -0.4)
        party_111 = p1.Perceptron("Party-gate 1 1 1", [0.6, 0.3, 0.2], -0.4)

        self.assertEqual(0, party_000.calc([0, 0, 0]))
        self.assertEqual(0, party_001.calc([0, 0, 1]))
        self.assertEqual(0, party_010.calc([0, 1, 0]))
        self.assertEqual(1, party_011.calc([0, 1, 1]))

        self.assertEqual(1, party_100.calc([1, 0, 0]))
        self.assertEqual(1, party_101.calc([1, 0, 1]))
        self.assertEqual(1, party_110.calc([1, 1, 0]))
        self.assertEqual(1, party_111.calc([1, 1, 1]))
