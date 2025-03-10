"""
This file has some hardcode parts that only work for the current setup format.
The hardcode parts are marked with #HARDCODE
"""
import pandas as pd
import numpy as np
import typing as t

from op._metrics import (
    Metrics, present_results, calculate_false_true_pred
)
from op.aux import save_csv_row


class TimeResults:
    """
    Return the different time stats
    """
    def __init__(self, time: t.Dict[str, float] | None) -> None:
        self.time = time

    def as_dict(self) -> t.Dict[str, float]:
        return {"NaN": "(No stats found)"} if self.time is None else self.time
    
    def format_time(self) -> t.Dict[str, float]:
        """Note, this only works for current setup format #HARDCODE"""
        data = self.as_dict()
        if 3 > len(data):
            return {} 
        return {
            "time agregation": data["Round 1 update"],
            "time threshold sel.": data["Round 1 threshold selection"],
            "time inference": data["Round 1 evaluation"],
            "time max local training": np.max(
                [data[d] for d in data.keys() if d.startswith("Round 1")]
            ),
            "time avg. local training": np.mean(
                [data[d] for d in data.keys() if d.startswith("Round 1")]
            ),
            "time std. local training": np.std(
                [data[d] for d in data.keys() if d.startswith("Round 1")]
            ),
        }


class Results:
    """
    Return the results of a binary classification. 

    Metrics use:
    * Precision
    * Recall
    * F1
    * Balance Accuracy
    """
    def __init__(self, tp: int, tn: int, fp: int, fn: int) -> None:
        self.tp, self.fn, self.tn, self.fp = tp, fn, tn, fp

        self.time_stats = self.add_time(None)
        self.precision = Metrics.precision(tp=tp, fp=fp)
        self.recall = Metrics.recall(tp=tp, fn=fn)
        self.f1 = Metrics.f1(tp=tp, fn=fn, fp=fp)
        self.balance_accuracy = Metrics.balance_accuracy(tp=tp, fn=fn, tn=tn, fp=fp)

    def add_time(self, time: t.Dict[str, float] | None) -> None:
        self.time_stats = TimeResults(time=time)

    def as_dict(self) -> t.Dict[str, t.Dict[str, t.Any]]:
        return {
            "Metrics": 
                {
                    "tp": self.tp,
                    "tn": self.tn,
                    "fp": self.fp,
                    "fn": self.fn,
                    "precision": self.precision,
                    "balance accuracy": self.balance_accuracy,
                    "recall": self.recall,
                    "f1": self.f1
                },
            "Times": self.time_stats.as_dict()
        }

    def __repr__(self) -> str:
        return present_results(self.as_dict())
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def as_csv(
        self, path: str | None, prefix: dict = {}
    ) -> None | pd.DataFrame: 
        """If None return the DataFrame instead"""
        time = self.time_stats.format_time()
        table = pd.DataFrame(
            prefix | {k : [v] for k, v in self.as_dict()["Metrics"].items()} | time
        )
        if path is None:
            return table 
        if len(table.columns) > 12 + len(prefix):  # Base only in current setup #HARDCODE
            save_csv_row(path, data=table)


def apply_metrics(
    pred_normal: t.List[int], 
    pred_abnormal: t.List[int], 
    times: t.Dict[str, float] | None = None
) -> Results:
    """
    Apply metrics to the test data.

    ```
    result = apply_metrics(test_normal, test_abnormal)
    # results.f1
    # results.recall
    # results.precision
    # results.time_stats
    ```
    """
    normal = calculate_false_true_pred(pred_normal, expected_value=0)
    abnormal = calculate_false_true_pred(pred_abnormal, expected_value=1)

    results = Results(
        tp=abnormal["True"],
        tn=normal["True"],
        fp=normal["False"],
        fn=abnormal["False"],
    )
    results.add_time(time=times)
    return results