
import models.ngram as ngram

import unittest


class NgramSetTestCase(unittest.TestCase):
    def test_add_item(self):
        ngramset = ngram.GramSet()
        self.assertSetEqual(ngramset.storage, set())

        ngramset + (1, 2)
        self.assertSetEqual(ngramset.storage, {(1, 2)})

    def test_add_set(self):
        ngramset = ngram.GramSet()
        self.assertSetEqual(ngramset.storage, set())

        ngramset + set([(1, 2), (2, 2)])
        self.assertSetEqual(ngramset.storage, {(1, 2), (2, 2)})

    def test_get_serialize(self) -> None:
        ngramset = ngram.GramSet()
        ngramset + set([(1, 2), (2, 2)])

        values = ngramset.get_as_serialize()
        self.assertEqual(len(values), 2)
        self.assertTrue(isinstance(values, list))
        self.assertTrue("(2, 2)" in values)
        self.assertTrue("(2, 2)" in values)

    def test_set_serialize(self) -> None:
        ngramset = ngram.GramSet()
        ngramset.set_as_serialize(["(1, 2)", "(2, 2)"])

        self.assertSetEqual(ngramset.storage, {(1, 2), (2, 2)})


class NgramTestCase(unittest.TestCase):
    def test_split_seq(self) -> None:
        x = [1, 2, 3, 4]

        self.assertListEqual(
            list(ngram.split_seq(x, 2)), 
            [(-1, 1), (1, 2), (2, 3), (3, 4), (4, -1)] 
        )
        self.assertListEqual(
            list(ngram.split_seq(x, 3)), 
            [(-1, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, -1)] 
        )
        self.assertListEqual(
            list(ngram.split_seq(x, 8)), [(-1, 1, 2, 3, 4, -1)] 
        )

    def test_get_weights(self) -> None:
        gram2 = ngram.NGram(2, thres=None)
        self.assertListEqual([], gram2.get_weights())

    def test_set_weights(self) -> None:
        gram2 = ngram.NGram(2, thres=None)
        values = ["(1, 2)", "(2, 2)"]
        gram2.set_weights(values)

        get_values = gram2.get_weights()
        self.assertEqual(len(get_values), 2)
        self.assertTrue("(2, 2)" in get_values)
        self.assertTrue("(1, 2)"in get_values)

    def test_fit(self) -> None:
        X = [[1], [2, 2, 4], [2, 6]]
        gram2 = ngram.NGram(2, thres=None)
        gram2.fit(X)

        self.assertSetEqual(
            {(-1, 1), (1, -1), (-1, 2), (2, 2), (2, 4), (4, -1), (2, 6), (6, -1)},
            gram2.gramset.storage
        )

    def test_score(self) -> None:
        X = [[1], [2, 2, 4], [2, 6]]
        gram2 = ngram.NGram(2, thres=None, div_max=True)
        gram2.fit(X)
        score = gram2.score([[2, 4], [4, 3], [2, 4]])

        self.assertListEqual(score, [0, 1, 0])

    def test_threshold(self) -> None:
        X = [[1], [2, 2, 4], [2, 6]]
        gram2 = ngram.NGram(2, thres=None)
        gram2.fit(X)
        gram2.set_threshold(
            X_abnormal=[[4, 3], [5, 8]], X_normal=[[2, 4], [2, 6]]
        )

        self.assertTrue(0 < gram2.threshold < 1)

    def test_predict(self) -> None:
        X = [[1], [2, 2, 4], [2, 6]]
        gram2 = ngram.NGram(2, thres=None)
        gram2.fit(X)
        gram2.set_threshold(
            X_abnormal=[[4, 3], [5, 8]], X_normal=[[2, 4], [2, 6]]
        )

        self.assertListEqual(
            gram2.predict([[4, 3], [5, 8], [2, 4], [2, 6]]), [1, 1, 0, 0]
        )

    def test_update_strategy(self) -> None:
        c_weights = [["(1, 2)", "(2, 2)"], [], ["(3, 4)"]]

        self.assertSetEqual(
            set(ngram.update_strategy(
                ngram.NGram(2, thres=None), clients_weights=c_weights
            )),
            {"(1, 2)", "(2, 2)", "(3, 4)"}
        )
