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


def present_results(results: dict) -> str:
    names = ["tp", "tn", "fp", "fn", "precision", "recall", "f1"]
    return "".join([f"||{name}: {results[name]:.2f}" for name in names]) + "||"


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