import models.known_events as ke
import unittest


class KnowEventsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.knownEvents = ke.KnownEvents()
    
    def test_set_weights(self) -> None:
        self.assertSetEqual(set(), self.knownEvents.events)
        self.knownEvents.set_weights(set(["a", "b", "c"]))
        self.assertSetEqual(set(["a", "b", "c"]), self.knownEvents.events)

    def test_get_weights(self) -> None:
        self.knownEvents.set_weights(set(["a", "V"]))
        self.assertSetEqual(
            self.knownEvents.get_weights(), set(["a", "V"])
        )

    def test_predict(self) -> None:
        sequences = [["a", "b", "a", "c"], ["a", "D"]]
        result = [0, 1]
        self.knownEvents.set_weights(set(["a", "b", "c"]))

        self.assertListEqual(result, self.knownEvents.predict(sequences))

    def test_fit(self) -> None:
        sequences = [["a", "b", "a", "c"], ["a", "D"]]
        self.assertEqual(self.knownEvents.fit(sequences), 4)
        self.assertListEqual(self.knownEvents.predict(sequences), [0, 0])
        self.assertSetEqual(self.knownEvents.get_weights(), set(["a", "b", "c", "D"]))

    def test_update_strategy(self) -> None:
        weights = [set(["a", "b", "c"]), set(["d", "a","c"])]
        
        self.assertSetEqual(
            ke.update_strategy(self.knownEvents, clients_weights=weights),
            set(["a", "b", "c", "d"])
        )