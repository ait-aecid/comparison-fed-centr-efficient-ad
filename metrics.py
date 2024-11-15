
import typing as t

from op._metrics import (
    Metrics, present_results, calculate_false_true_pred
)


class Results:
    """
    Return the results of a binary classification.

    Metrics use:
    * Precision
    * Recall
    * F1
    """
    def __init__(self, tp: int, tn: int, fp: int, fn: int) -> None:
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

        self.precision = Metrics.precision(tp=tp, fp=fp)
        self.recall = Metrics.recall(tp=tp, fn=fn)
        self.f1 = Metrics.f1(tp=tp, fn=fn, fp=fp)

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }

    def __repr__(self) -> str:
        return present_results(self.as_dict())
    
    def __str__(self) -> str:
        return self.__repr__()


def apply_metrics(
    pred_normal: t.List[int], pred_abnormal: t.List[int]
) -> Results:
    """
    Apply metrics to the test data.

    ```
    result = apply_metrics(test_normal, test_abnormal)
    # results.f1
    # results.recall
    # results.precision
    ```
    """
    normal = calculate_false_true_pred(pred_normal, expected_value=0)
    abnormal = calculate_false_true_pred(pred_abnormal, expected_value=1)

    return Results(
        tp=abnormal["True"],
        tn=normal["True"],
        fp=normal["False"],
        fn=abnormal["False"],
    )