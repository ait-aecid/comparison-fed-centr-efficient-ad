"""
NGRAM method from: https://dl.acm.org/doi/pdf/10.1145/3660768

Pseudo code:
------------
X -> [(sequence, next_event) ... ]
score = max(ngram[sequence]) == next_event
is_anomaly = if score >= threshold

Federated approach: 
-------------------
Not added yet
"""
from models._imodel import Model

from typing import Any, List, Tuple, Self
import numpy as np


def split_seq(X: List[List[Any]], sep: int) -> List[List[Any]]:
    split = []
    for xi in X:
        x_ = [-1 for _ in range(sep)] + xi
        for j in range(0, len(x_) - sep):
            split.append([x_[j: j + sep], x_[j + sep]])

    return split



class GramMatrix:
    def __init__(self) -> None:
        self.storage = {}
        self.vocab = []

    def get_values(self) -> Tuple[List[Any], List[Any], List[List[int]]]:
        return [self.vocab, list(self.storage.keys()), list(self.storage.values())]

    @classmethod
    def from_values(
        cls, values: Tuple[List[Any], List[Any], List[List[int]]]
    ) -> Self:
        
        matrix = cls()
        matrix.vocab = values[0]
        matrix.storage = {k:v for k, v in zip(values[1], values[2])}
        return matrix

    def __getitem__(self, idx: Any) -> List[Any]:  
        return self.storage[idx]
    
    def __contains__(self, idx: Any) -> bool:
        return idx in self.storage.keys()

    def __update_vocab(self, value: Any) -> None:
        self.vocab.append(value)
        for k in self.storage.keys():
            self.storage[k].append(0)
    
    def update(self, idx: Any, value: Any) -> None:
        if idx not in self:
            self.storage[idx] = [0 for _ in self.vocab]
        if value not in self.vocab:
            self.__update_vocab(value) 

        self.storage[idx][self.vocab.index(value)] += 1


class NGram(Model):
    def __init__(self, n: int) -> None:
        self.n = n
        self.matrix = GramMatrix()

    def get_weights(self) -> List[Any]:
        return self.matrix.get_values()
    
    def set_weights(self, weights: List[Any]) -> None:
        self.matrix = self.matrix.from_values(weights) 

    def fit(self, X: List[List[Any]]) -> float:
        for [seq, e] in split_seq(X, sep=self.n):
            self.matrix.update(str(seq), e)
        return 0.
    
    def predict(self, X: List[List[Any]]) -> List[int]:
        results = []
        for xi in X:
            errors_count = 0
            for seq, e in split_seq([xi], sep=self.n):
                if str(seq) not in self.matrix:
                    errors_count += 1
                else:
                    pred = self.matrix.vocab[np.argmax(self.matrix[str(seq)])]
                    if pred != e:
                        errors_count += 1
            results.append(errors_count)
        return results  # TODO: missing thershold

    