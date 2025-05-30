from metrics import apply_metrics
import op._metrics as m

import unittest


class MetricsTestCase(unittest.TestCase):
    def test_calculate_pred(self) -> None:
        self.assertDictEqual(
            {"True": 5, "False": 0},
            m.calculate_false_true_pred([0, 0, 0, 0, 0], 0)
        )
        self.assertDictEqual(
            {"True": 0, "False": 5},
            m.calculate_false_true_pred([0, 0, 0, 0, 0], 1)
        )
        self.assertDictEqual(
            {"True": 3, "False": 2},
            m.calculate_false_true_pred([0, 0, 1, 1, 1], 1)
        )

    def test_results_perfect(self) -> None:
        result = apply_metrics(
            pred_normal=[0, 0, 0], pred_abnormal=[1, 1, 1]
        )

        self.assertEqual(result.f1, 1.)
        self.assertEqual(result.recall, 1.)
        self.assertEqual(result.precision, 1.)
        self.assertEqual(result.balance_accuracy, 1.)
        self.assertEqual(result.tp, 3)
        self.assertEqual(result.tn, 3)
        self.assertEqual(result.fn, 0)
        self.assertEqual(result.fp, 0)        

    def test_results_not_perfect(self) -> None:
        result = apply_metrics(
            pred_normal=[1, 1, 1], pred_abnormal=[1, 1, 1]
        )

        self.assertAlmostEqual(result.f1, 0.66, delta=0.01)
        self.assertEqual(result.recall, 1.)
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.balance_accuracy, 0.5)
        self.assertEqual(result.tp, 3)
        self.assertEqual(result.tn, 0)
        self.assertEqual(result.fn, 0)
        self.assertEqual(result.fp, 3)        

    def test_results_dict(self) -> None:
        result = apply_metrics(
            pred_normal=[0, 0, 0], pred_abnormal=[1, 1, 1]
        )

        self.assertDictEqual({
            "Metrics": {
                "f1": 1.,
                "recall": 1.,
                "precision": 1.,
                "balance accuracy": 1.,
                "tp": 3,
                "tn": 3,
                "fn": 0,
                "fp": 0,
            }, "Times": {"NaN": "(No stats found)"}
            },
            result.as_dict()
        )

    def test_results_dict_with_time(self) -> None:
        result = apply_metrics(
            pred_normal=[0, 0, 0], 
            pred_abnormal=[1, 1, 1],
            times={"a": 2, "b": 1}
        )

        self.assertDictEqual({
            "Metrics": {
                "f1": 1.,
                "recall": 1.,
                "precision": 1.,
                "balance accuracy": 1.,
                "tp": 3,
                "tn": 3,
                "fn": 0,
                "fp": 0,
            }, "Times": {"a": 2, "b": 1}
            },
            result.as_dict()
        )

    def test_as_csv(self) -> None:
        result = apply_metrics(
            pred_normal=[0, 0, 0], 
            pred_abnormal=[1, 1, 1],
        )
        output = result.as_csv(None)
        self.assertListEqual(
            list(output.columns), 
            ["tp", "tn", "fp", "fn", "precision", "balance accuracy", "recall", "f1"]
        )
        expected = {
            'tp': [3], 'tn': [3], 'fp': [0], 'fn': [0], 'precision': [1.0], 
            'balance accuracy': [1.0], 'recall': [1.0], 'f1': [1.0]
        }
        self.assertDictEqual(expected, output.to_dict("list"))
        

    def test_as_csv_with_prefix(self) -> None:
        result = apply_metrics(
            pred_normal=[0, 0, 0], 
            pred_abnormal=[1, 1, 1],
        )
        output = result.as_csv(None, prefix={"a": [1], "b": [2]})
        self.assertListEqual(
            list(output.columns), 
            ["a", "b", "tp", "tn", "fp", "fn", "precision", "balance accuracy", "recall", "f1"]
        )
        expected = {
            'tp': [3], 'tn': [3], 'fp': [0], 'fn': [0], 'precision': [1.0], 
            'balance accuracy': [1.0], 'recall': [1.0], 'f1': [1.0], "a": [1], "b": [2]
        }
        self.assertDictEqual(expected, output.to_dict("list"))