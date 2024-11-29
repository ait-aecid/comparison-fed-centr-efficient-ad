"""
Lenght method from: https://dl.acm.org/doi/pdf/10.1145/3660768

Pseudo code:
------------
is_abnormal = not (min_value <= len(seq) <= max_value)

Federated approach: 
-------------------
We take the min and max value from all the clients
"""
from models._imodel import Model

import typing as t


def _format_sequence(func):
    def infunc(self, sequence):
        n = sequence if isinstance(sequence, int) else len(sequence)
        return func(self, n)
    return infunc


class _RangeClass:
    def __init__(self) -> None:
        self.min_length = 1e10
        self.max_length = -1

    @_format_sequence
    def __gt__(self, sequence: t.List[t.Any] | int) -> bool:
        return self.max_length > sequence

    @_format_sequence
    def __ge__(self, sequence: t.List[t.Any] | int) -> bool:
        return self > sequence or self.max_length == sequence

    @_format_sequence
    def __lt__(self, sequence: t.List[t.Any] | int) -> bool:
        return self.min_length < sequence 

    @_format_sequence
    def __le__(self, sequence: t.List[t.Any] | int) -> bool:
        return self < sequence or self.min_length == sequence

    @_format_sequence
    def __contains__(self, sequence: t.List[t.Any]) -> bool:
        return self <= sequence and self >= sequence


class LengthDetection(Model, _RangeClass):
    def update(self, value: int) -> None:
        [min_, max_] = self.get_weights()
        self.set_weights([
            value if value < min_ else min_,
            value if value > max_ else max_,
        ])

    def set_weights(self, weights: t.List[t.Any]) -> None:
        [self.min_length, self.max_length] = weights

    def get_weights(self) -> t.List[t.Any]:
        return [self.min_length, self.max_length]

    def fit(self, X: t.List[t.List[t.Any]]) -> float:
        for xi in X:
            if xi not in self:
                self.update(len(xi))
        return 0.
    
    def predict(self, X: t.List[t.List[t.Any]]) -> t.List[int]:
        return [xi not in self for xi in X]


def update_strategy(
    server_model: LengthDetection, clients_weights: t.List[t.List[t.Any]]
) -> t.List[t.Any]:

    for [min_length, max_length] in clients_weights:
        server_model.update(min_length)
        server_model.update(max_length)

    return server_model.get_weights()