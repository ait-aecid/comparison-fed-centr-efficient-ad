
"""
NGRAM method from: https://dl.acm.org/doi/pdf/10.1145/3660768

Pseudo code:
------------
X -> [(sequence, next_event) ... ]
score = following equation from the paper 
is_anomaly = if score >= threshold

Federated approach: 
-------------------
Join the ngram tables of each client
"""
from models._thresholds import (
    supervised_threshold_selection, apply_threshold, Thresholds
)
from models._imodel import Model

from typing import Any, List, Tuple, Set, Iterator, Self
from tqdm import tqdm
import torch


def split_seq(x: List[Any], sep: int) -> Iterator[Tuple[Any]]:
    x_ = [-1] + x + [-1]
    if len(x_) < sep:
        yield tuple(x_)

    for j in range(0, len(x_) - sep + 1):
        yield tuple(x_[j: j + sep])


class GramSet:
    def __init__(self) -> None:
        self.storage = set()

    def __len__(self) -> int:
        return len(self.storage)

    def __contains__(self, idx: Any) -> bool:
        return idx in self.storage

    def __add__(self, idx: Any | Set[Any]) -> Self:
        if isinstance(idx, set):
            self.storage = self.storage.union(idx)
        else:
            self.storage.add(idx)
        return self

    def get_as_serialize(self) -> List[str]:
        return [str(v) for v in self.storage]

    def set_as_serialize(self, data: List[str]) -> None:
        for v in data:
            self + eval(v)


class NGram(Model):
    def __init__(self, n: int, thres: float | None) -> None:
        super().__init__(name=f"{n}-Gram", thres=thres)
        self.get_thres = self.threshold is None 
        self.gramset = GramSet()
        self.n = n

    def get_weights(self) -> List[Any]:
        return self.gramset.get_as_serialize()
    
    def set_weights(self, weights: List[Any]) -> None:
        self.gramset.set_as_serialize(weights)

    def fit(self, X: List[List[Any]]) -> float:
        for x in tqdm(X):
            for seq in split_seq(x, sep=self.n):
                self.gramset + seq
        return len(self.gramset)

    def score(self, X: List[List[Any]]) -> List[int]:
        results, mn_max = [], 10e-10
        for x in tqdm(X):
            results.append(0)
            for seq in split_seq(x, sep=self.n):
                if seq not in self.gramset:
                    results[-1] += 1
            
            nmax = 1 if self.n >= len(seq) else  self.n * (len(seq) - self.n / 2)
            results[-1] /= nmax
            mn_max = max(mn_max, results[-1])

        return (torch.Tensor(results) / mn_max).detach().tolist()

    def set_threshold(
        self, X_normal: List[List[Any]], X_abnormal: List[List[Any]]
    ) -> None:
        if self.get_thres:
            self.threshold = supervised_threshold_selection(
                score_normal=self.score(X_normal),
                score_abnormal=self.score(X_abnormal),
            )

    def predict(self, X: List[List[Any]]) -> List[int]:
        return apply_threshold(self.score(X), threshold=self.threshold)


class NGram2(NGram):
    def __init__(self, thres: Thresholds = Thresholds()) -> None:
        super().__init__(n=2, thres=thres.gram2)


class NGram3(NGram):
    def __init__(self, thres: Thresholds = Thresholds()) -> None:
        super().__init__(n=3, thres=thres.gram3)
 

def update_strategy(
    server_model: NGram, clients_weights: List[List[Any]]
) -> List[Any]:

    for client_weights in clients_weights:
        ngram_ = NGram(server_model.n, thres=0)
        ngram_.set_weights(client_weights)
        server_model.gramset + ngram_.gramset.storage

    return server_model.get_weights()