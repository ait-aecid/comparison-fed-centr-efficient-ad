

from models.lenght_detection import LengthDetection, update_strategy

import unittest


class LengthDetectionTestCase(unittest.TestCase):
    def test_larger_smaller_than(self) -> None:
        length = LengthDetection()
        length.min_length = 2
        length.max_length = 3

        self.assertFalse(length > [1, 2, 3, 4])
        self.assertTrue(length > [1])
        self.assertTrue(length < [1, 2, 3, 4])
        self.assertFalse(length < [1])

    def test_inside_length(self) -> None:
        length = LengthDetection()
        length.min_length = 2
        length.max_length = 4

        self.assertTrue([1, 2] in length)
        self.assertTrue([1, 2, 3] in length)
        self.assertTrue([1, 2, 3, 4] in length)
        self.assertFalse([] in length)
        self.assertFalse([1, 2, 3, 4, 5] in length)

    def test_get_weights(self) -> None:
        length = LengthDetection()
        length.min_length = 2
        length.max_length = 6

        self.assertListEqual(length.get_weights(), [2, 6])

    def test_set_weights(self) -> None:
        length = LengthDetection()
        length.set_weights([1, 5])

        self.assertListEqual(length.get_weights(), [1, 5])

    def test_fit(self) -> None:
        length = LengthDetection()
        sequences = [["a", "b", "a", "c"], ["a", "D"], ["a", "b", "C"]]
        length.fit(sequences)

        self.assertListEqual(length.get_weights(), [2, 4])

    def test_predict(self) -> None:
        length = LengthDetection()
        sequences = [["a", "b", "a", "c"], ["a", "D"], ["a", "b", "C"]]
        length.fit(sequences)
        print(length.get_weights())

        test = [["a"], [1, 2, 3], [1, 2, 3, 4, 5, 6]]
        self.assertListEqual(length.score(test), [1, 0, 1])

    def test_update_strategy(self) -> None:
        length = LengthDetection()
        length.set_weights(length.get_weights())
        weights = [[1, 2], [4, 8], [0, 3]]

        self.assertListEqual(
            update_strategy(length, clients_weights=weights), [0, 8]
        )