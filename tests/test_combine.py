from models.combine import Combine, update_strategy

from models.lenght_detection import LengthDetection
from models.edit_distance import EditDistance
from models.known_events import KnownEvents
from models.ecvc import ECVC

from models.known_events import update_strategy as known_update
from models.ecvc import update_strategy as ecvc_update

import unittest


class CombineTestCase(unittest.TestCase):
    def test_fit(self) -> None:
        known_events = KnownEvents()
        length_detection = LengthDetection()
        combine = Combine([known_events, length_detection])

        X = [[1, 2, 3], [1, 1]]
        combine.fit(X)

        self.assertSetEqual({1, 2, 3}, set(known_events.get_weights()))
        self.assertSetEqual(
            {2, 3}, set(length_detection.get_weights())
        )

    def test_predict(self) -> None:
        known_events = KnownEvents()
        edit_distance = LengthDetection()
        combine = Combine([known_events, edit_distance])

        X = [[1, 2, 3], [1, 1]]
        combine.fit(X)
        result = combine.predict([[1, 2, 3], [3, 2], [0, 5, 8, 9, 11]])

        self.assertListEqual([0, 0, 1], result)

    def test_predict_with_setthreshold(self) -> None:
        known_events = KnownEvents()
        edit_distance = EditDistance()
        combine = Combine([known_events, edit_distance])

        X = [[1, 2, 3], [1, 1]]
        combine.fit(X)
        combine.set_threshold(
            X_normal=[[1, 2, 3], [3, 2]], X_abnormal=[[0, 5, 8, 9, 11]]
        )
        result = combine.predict([[1, 2, 3], [3, 2], [0, 5, 8, 9, 11]])

        self.assertAlmostEqual(edit_distance.threshold, 0.6676676869392395, delta=0.01)        
        self.assertListEqual([0, 0, 1], result)

    def test_get_weights(self) -> None:
        known_events = KnownEvents()
        ecvc = ECVC()
        combine = Combine([known_events, ecvc])

        X = [[1, 2, 3], [1, 1]]
        combine.fit(X)
        weights = combine.get_weights()

        self.assertEqual(len(weights), 2)
        self.assertSetEqual(set(eval(weights[0])), {1, 2, 3})
        ecvc_weights = eval(weights[1])
        self.assertEqual(len(ecvc_weights), 2)
        self.assertTrue([0, 1, 1, 1] in ecvc_weights)
        self.assertTrue([0, 2, 0, 0] in ecvc_weights)

    def test_set_weights(self) -> None:
        known_events = KnownEvents()
        ecvc = ECVC()
        combine = Combine([known_events, ecvc])
        combine.set_weights(["[1, 2, 3]", "[[0, 1, 1, 1], [0, 2, 0, 0]]"])
        weights = combine.get_weights()

        self.assertEqual(len(weights), 2)
        self.assertSetEqual(set(eval(weights[0])), {1, 2, 3})
        ecvc_weights = eval(weights[1])
        self.assertEqual(len(ecvc_weights), 2)
        self.assertTrue([0, 1, 1, 1] in ecvc_weights)
        self.assertTrue([0, 2, 0, 0] in ecvc_weights)

    def test_update_strategy(self) -> None:
        known_events = KnownEvents()
        ecvc = ECVC()
        combine = Combine(
            models=[known_events, ecvc],
            update_funcs=[known_update, ecvc_update],
        )

        client_weights = [["[1]", "[[0, 1, 1, 1]]"], ["[2, 3]", "[[0, 2, 0, 0]]"]]
        update_strategy(combine, clients_weights=client_weights)
        weights = combine.get_weights()

        self.assertEqual(len(weights), 2)
        self.assertSetEqual(set(eval(weights[0])), {1, 2, 3})
        ecvc_weights = eval(weights[1])
        self.assertEqual(len(ecvc_weights), 2)
        self.assertTrue([0, 1, 1, 1] in ecvc_weights)
        self.assertTrue([0, 2, 0, 0] in ecvc_weights)