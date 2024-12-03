
import models.ngram as ngram

import unittest


class SerializationTestCasTestCase(unittest.TestCase):
    def test_convert(self) -> None:
        weights = [
            ["a", "b", "c"],
            ["aa", "ab", "ac", "bc"],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]]
        ]
        self.assertListEqual(
            ngram.Serilize.convert(weights),
            [
                "a", "b", "c", "<END>", "aa", "ab", "ac", "bc", "<END>",
                1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
            ]            
        )

    def test_convert_empty(self) -> None:
        self.assertListEqual(
            ngram.Serilize.convert([[], [], []]), ["<END>", "<END>"]            
        )

    def test_inverse(self) -> None:
        weights = [
            ["a", "b", "c"],
            ["aa", "ab", "ac", "bc"],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]]
        ]
        data = ngram.Serilize.inverse([
            "a", "b", "c", "<END>", "aa", "ab", "ac", "bc", "<END>",
            1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
        ])

        self.assertListEqual(weights, data)

    def test_inverse_empty(self) -> None:
        self.assertListEqual(
            ngram.Serilize.inverse( ["<END>", "<END>"]),[[], [], []]            
        )


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
        print(self.matrix.storage)
    
        for idx in self.matrix.storage.keys():
            print(self.matrix[idx])
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

        self.assertListEqual(weights, ["<END>", "<END>"])

    def test_set_weights(self) -> None:
        weights = [
            "a", "b", "c", "<END>", "aa", "ab", "ac", "bc", "<END>",
            1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
        ]
        gram2 = ngram.NGram(2)
        gram2.set_weights(weights)
        result = gram2.get_weights()

        self.assertListEqual(
            [
                "a", "b", "c", "<END>", "aa", "ab", "ac", "bc", "<END>",
                1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
            ], result
        )            

    def test_fit(self) -> None:
        X = [["a", "b", "b", "b", "c"], ["c"], ["b", "b"]]
        gram2 = ngram.NGram(2)
        gram2.fit(X)

        vocab = gram2.matrix.vocab
        comb = gram2.matrix.get_combs()
        expected = [
            "['a', 'b']", "[-1, 'b']", '[-1, -1]', "[-1, 'a']", "['b', 'b']"
        ]
        self.assertSetEqual({"a", "b", "c"}, set(vocab))
        self.assertSetEqual(set(expected), set(comb))
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
        self.assertEqual(result, [0, 0, 2, 1])

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

    def test_set_threshold_no_train(self) -> None:
        gram2 = ngram.NGram(2)

        gram2.set_threshold(
            X_normal=[["a", "b"], ["b", "b"], ["a", "c"]],
            X_abnormal=[["z", "a"]],
        )
        self.assertEqual(gram2.threshold, 2)

    def test_update_strategy(self) -> None:
        client_weights = [
            ["a", "b", "<END>", "aa", "ba", "<END>", "0", "1", "1", "0"],
            ["c", "a", "<END>","ca", "aa", "<END>", "0", "1", "1", "2"],
        ]
        gram2 = ngram.NGram(2)
        weights = ngram.update_strategy(
            server_model=gram2, clients_weights=client_weights
        )
        [vocab, combs, values] = ngram.Serilize.inverse(weights)

        self.assertSetEqual(set(vocab), {"a", "b", "c"})
        self.assertSetEqual(set(combs), {"aa", "ba", "ca"})

        gram2 = ngram.NGram(2)
        gram2.set_weights(weights)

        values_aa = gram2.matrix["aa"]
        self.assertEqual(2, values_aa[vocab.index("a")]) 
        self.assertEqual(1, values_aa[vocab.index("b")]) 
        self.assertEqual(1, values_aa[vocab.index("c")]) 
        values_aa = gram2.matrix["ba"]
        self.assertEqual(1, values_aa[vocab.index("a")]) 
        self.assertEqual(0, values_aa[vocab.index("b")]) 
        self.assertEqual(0, values_aa[vocab.index("c")]) 
        values_aa = gram2.matrix["ca"]
        self.assertEqual(1, values_aa[vocab.index("a")]) 
        self.assertEqual(0, values_aa[vocab.index("b")]) 
        self.assertEqual(0, values_aa[vocab.index("c")]) 