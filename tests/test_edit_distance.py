
import models.edit_distance as edit

import unittest


class EditDistanceTestCase(unittest.TestCase):
    def test_get_weights(self) -> None:
        editd = edit.EditDistance()
        editd.sequences = set([(1, 2, 3), (4, 3, 2)]) 

        self.assertTrue(isinstance(w := editd.get_weights(), list))
        self.assertSetEqual(set(["(1, 2, 3)", "(4, 3, 2)"]), set(w))

    def test_set_weights(self) -> None:
        editd = edit.EditDistance()
        editd.set_weights(["(1, 2, 3)", "(4, 3, 2)"])

        self.assertTrue(isinstance(w := editd.get_weights(), list))
        self.assertSetEqual(set(["(1, 2, 3)", "(4, 3, 2)"]), set(w))

    def test_fit(self) -> None:
        X = [[1, 2, 3], [4, 5], [1, 2, 3]]
        editd = edit.EditDistance()
        editd.fit(X)

        self.assertTrue(isinstance(w := editd.get_weights(), list))
        self.assertSetEqual(set(["(1, 2, 3)", "(4, 5)"]), set(w))

    def test_leve_distance(self) -> None:
        self.assertEqual(0, edit.levenshtein_distance((1, 2, 3), (1, 2, 3)))
        self.assertEqual(1, edit.levenshtein_distance((1, 3, 3), (1, 5, 3)))
 
        self.assertEqual(2, edit.levenshtein_distance(
            (1, 3, 3), (1, 5, 5), score_cutoff=1
        ))
        self.assertEqual(1, edit.levenshtein_distance(
            (1, 3, 3), (1, 5, 5), score_cutoff=0.9
        ))

    def test_score(self) -> None:
        X = [[1, 2, 3], [4, 5], [1, 2, 3]]
        editd = edit.EditDistance()
        editd.fit(X)

        self.assertListEqual(editd.score(
            [[1, 2, 3], [5, 5], [5, 3], [5, 5]]
        ), [0, 1, 1, 1])

    def test_score_empty(self) -> None:
        editd = edit.EditDistance()

        self.assertListEqual(editd.score([[1, 2, 3], [5, 5], [5, 3]]), [-1, -1, -1])

    def test_set_threshold(self) -> None:
        X = [[1, 2, 3], [4, 5], [1, 2, 3]]
        editd = edit.EditDistance()
        editd.fit(X)
        editd.set_threshold(
            X_normal=[[1, 2, 3], [5, 5]],
            X_abnormal=[[5, 3]]
        )
        
        self.assertTrue(1 <= editd.threshold <= 2)

    def test_predict(self) -> None:
        X = [[1, 2, 3], [4, 5], [1, 2, 3]]
        editd = edit.EditDistance()
        editd.fit(X)
        editd.set_threshold(
            X_normal=[[1, 2, 3], [5, 5]],
            X_abnormal=[[5, 3]]
        )

        self.assertListEqual(
            editd.predict([[1, 2, 3], [5, 3], [5, 5]]), [0, 0, 0]
        )        

    def test_update_strategy(self) -> None:
        X = [[1, 2, 3], [4, 5], [1, 2, 3]]
        editd1 = edit.EditDistance()
        editd1.fit(X)
        X = [[1, 2, 3], [2, 2]]
        editd2 = edit.EditDistance()
        editd2.fit(X)
        editd3 = edit.EditDistance()

        weights = edit.update_strategy(
            edit.EditDistance(), 
            clients_weights=[
                editd1.get_weights(), editd2.get_weights(), editd3.get_weights()
            ]
        )

        self.assertTrue(isinstance(weights, list))
        self.assertSetEqual(set(["(1, 2, 3)", "(4, 5)", "(2, 2)"]), set(weights))

