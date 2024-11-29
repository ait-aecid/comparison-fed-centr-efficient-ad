
import models.ngram as ngram

import unittest


class GramMatrixTestCase(unittest.TestCase):
    def setUp(self) -> None:
        values = [
            ["a", "b", "c"],
            ["aa", "ab", "ac", "bc"],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]]
        ]
        self.matrix = ngram.GramMatrix.from_values(values)

    def test_get_values(self) -> None:
        matrix = ngram.GramMatrix()
        values = matrix.get_values()

        self.assertListEqual(values[0], [])
        self.assertListEqual(values[1], [])
        self.assertListEqual(values[2], [])

    def test_contains(self) -> None:
        self.assertTrue("aa" in self.matrix)
        self.assertFalse("zz" in self.matrix)

    def test_from_values(self) -> None:
        values = self.matrix.get_values()

        self.assertListEqual(values[0], ["a", "b", "c"])
        self.assertListEqual(values[1], ["aa", "ab", "ac", "bc"])
        self.assertListEqual(values[2], [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]])

    def test_get_items(self) -> None:
        self.assertListEqual(self.matrix["aa"], [1, 0, 0])
        self.assertListEqual(self.matrix["bc"], [1, 0, 1])

    def test_add_items(self) -> None:
        self.assertListEqual(self.matrix["aa"], [1, 0, 0])
        self.matrix.update("aa", "a")
        self.assertListEqual(self.matrix["aa"], [2, 0, 0])

    def test_add_new_item(self) -> None:
        self.matrix.update("cd", "a")
        self.assertListEqual(self.matrix["cd"], [1, 0, 0])

    def test_add_new_value(self) -> None:
        self.matrix.update("aa", "z")
        self.assertListEqual(self.matrix["aa"], [1, 0, 0, 1])
    
        for idx in self.matrix.storage.keys():
            self.assertEqual(4, len(self.matrix[idx]))


class NGramTestCase(unittest.TestCase):
    def test_split_seq(self) -> None:
        X = [[1, 2, 3], [5]]
        
        self.assertListEqual(
            [
                [[-1, -1], 1],
                [[-1, 1], 2],
                [[1, 2], 3],
                [[-1, -1], 5]
            ], ngram.split_seq(X, sep=2)
        )
        self.assertListEqual(
            [
                [[-1, -1, -1], 1],
                [[-1, -1, 1], 2],
                [[-1, 1, 2], 3],
                [[-1, -1, -1], 5]
            ], ngram.split_seq(X, sep=3)
        )

    def test_get_weights(self) -> None:
        gram2= ngram.NGram(2)
        weights = gram2.get_weights()

        self.assertListEqual(weights[0], [])
        self.assertListEqual(weights[1], [])
        self.assertListEqual(weights[2], [])

    def test_set_weights(self) -> None:
        weights = [
            ["a", "b", "c"],
            ["aa", "ab", "ac", "bc"],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]]
        ]
        gram2 = ngram.NGram(2)
        gram2.set_weights(weights)
        result = gram2.get_weights()

        self.assertListEqual(weights[0], result[0])
        self.assertListEqual(weights[1], result[1])
        self.assertListEqual(weights[2], result[2])

    def test_fit(self) -> None:
        X = [["a", "b", "b", "b", "c"], ["c"], ["b", "b"]]
        gram2 = ngram.NGram(2)
        gram2.fit(X)
        result = gram2.get_weights()

        expected = [
            "['a', 'b']", "[-1, 'b']", '[-1, -1]', "[-1, 'a']", "['b', 'b']"
        ]
        self.assertSetEqual({"a", "b", "c"}, set(result[0]))
        self.assertSetEqual(set(expected), set(result[1]))
        self.assertListEqual(gram2.matrix['[-1, -1]'], [1, 1, 1])
        self.assertListEqual(gram2.matrix['[-1, -1]'], [1, 1, 1])
        self.assertEqual(gram2.matrix["['b', 'b']"][gram2.matrix.vocab.index("b")], 1)
        self.assertEqual(gram2.matrix["['b', 'b']"][gram2.matrix.vocab.index("a")], 0)
        self.assertEqual(gram2.matrix["['b', 'b']"][gram2.matrix.vocab.index("c")], 1)

    def test_score(self) -> None:
        X = [["a", "b", "b", "b", "c"], ["c"], ["b", "b"]]
        gram2 = ngram.NGram(2)
        gram2.fit(X)

        result = gram2.score([["a", "b"], ["b", "b"], ["z", "a"], ["a", "c"]])
        self.assertEqual(result, [0, 1, 2, 1])

    def test_set_threshold(self) -> None:
        X = [["a", "b", "b", "b", "c"], ["c"], ["b", "b"]]
        gram2 = ngram.NGram(2)
        gram2.fit(X)

        gram2.set_threshold(
            X_normal=[["a", "b"], ["b", "b"], ["a", "c"]],
            X_abnormal=[["z", "a"]],
        )
        self.assertTrue(1 <= gram2.threshold <= 2)

    def test_set_predict(self) -> None:
        X = [[1, 2, 2, 3, 4], [3], [4, 4]]
        gram2 = ngram.NGram(2)
        gram2.fit(X)
        gram2.set_threshold(
            X_normal=[[1, 2], [4, 4], [1, 3]],
            X_abnormal=[[10, 1]],
        )

        x_test = [[1, 2], [4, 4], [10, 1], [1, 3]]
        self.assertListEqual(gram2.predict(x_test), [0, 0, 1, 0])


