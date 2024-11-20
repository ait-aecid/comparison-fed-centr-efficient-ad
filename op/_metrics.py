from tabulate import tabulate

import typing as t


class Metrics:
    @staticmethod
    def __not_zero(x: float) -> float:
        return x if x != 0 else 1e-10

    @staticmethod
    def precision(tp: int, fp: int) -> float:
        return tp / Metrics.__not_zero(tp + fp)

    @staticmethod
    def recall(tp: int, fn: int) -> float:
        return tp / Metrics.__not_zero(tp + fn)

    @staticmethod
    def f1(tp: int, fn: int, fp: int) -> float:
        return 2 * tp / Metrics.__not_zero(2 * tp + fp + fn)

    @staticmethod
    def balance_accuracy(tp: int, fn: int, tn: int, fp: int) -> float:
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def present_results(results: t.Dict[str, t.Dict[str, float]]) -> str:
    msg = ""
    for key, sub_dict in results.items():
        msg += f"{key}\n"
        msg += f"{tabulate(
            sub_dict.items(), ["Metrics", "Value"], tablefmt="rounded_grid"
        )}\n\n" 
    return msg


def calculate_false_true_pred(
    preds: t.List[int], expected_value: int
) -> t.Dict[str, int]:
    
    results = {"True": 0, "False": 0}
    for pred in preds:
        if pred == expected_value:
            results["True"] += 1
        else:
            results["False"] += 1 

    return results 