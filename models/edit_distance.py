"""
Edit Distance method from: https://dl.acm.org/doi/pdf/10.1145/3660768

Pseudo code:
------------
score = LevensheteinDistance(train_seqs, seq) 
is_abnormal = score >= threshold 

Federated approach: 
-------------------
for weigth in client_weights:
    server_weights.extend(weight)
"""
from models._thresholds import (
    supervised_threshold_selection, apply_threshold, Thresholds
)
from models._imodel import Model

from typing import Any, List, Tuple
import Levenshtein as Leve
from tqdm import tqdm 
import math


def levenshtein_distance(
    seq_1: Tuple[int], seq_2: Tuple[int], score_cutoff: float | None = None
) -> float | int:
    return Leve.distance(s1=seq_1, s2=seq_2, score_cutoff=score_cutoff)


class EditDistance(Model):
    """
    It use the Levenshtein distance to compare the sequences.
    """
    def __init__(
        self, thres: Thresholds = Thresholds(), do_cutoff: bool = False
    ) -> None:
        super().__init__(name="EditDistance", thres=thres.edit)
        self.get_thres = self.threshold is None
        self.sequences = set()
        self.do_cutoff = do_cutoff

    def __len__(self) -> int:
        n = 0
        for seq in self.sequences:
            n += len(seq)
        return n

    def __add__(self, value: List[Any]) -> None:
        self.sequences.add(tuple(value))

    def set_weights(self, weights: List[Any]) -> None:
        self.sequences = set([eval(seq) for seq in weights])

    def get_weights(self) -> List[Any]:
        return [str(seq) for seq in self.sequences]
    
    def set_threshold(
        self, X_normal: List[List[Any]], X_abnormal: List[List[Any]]
    ) -> None:
        if self.get_thres:
            self.threshold = supervised_threshold_selection(
                score_normal=self.score(X_normal),
                score_abnormal=self.score(X_abnormal),
            )

    def score(self, X: List[List[Any]]) -> List[int]:
        if len(self.sequences) == 0:
            return [-1 for _ in X]

        scores, cache = [], {}
        for xi in tqdm(X):
            xi_ = tuple(xi)
            if xi_ in cache.keys():
                scores.append(cache[xi_])
            else:
                min_dist = 2
                for seq in self.sequences:
                    norm = float(max(len(seq), len(xi_)))
                    cutoff = math.floor(norm * min_dist) if self.do_cutoff else None
                    dist = levenshtein_distance(seq, xi_, score_cutoff=cutoff) / norm
                    if dist < min_dist:
                        min_dist = dist

                scores.append(min_dist)
                cache[xi_] = scores[-1]
        return scores

    def fit(self, X: List[List[Any]]) -> float:
        self.sequences = set()
        for xi in X:
            self + xi
        print(len(self.sequences), len(X))
        return len(self)
    
    def predict(self, X: List[List[Any]]) -> List[int]:
        return apply_threshold(score=self.score(X), threshold=self.threshold)


def update_strategy(
    server_model: EditDistance, clients_weights: List[List[Any]]
) -> List[Any]:
    weights = set()
    for client in clients_weights:
        server_model.set_weights(client)
        weights = weights.union(server_model.sequences)
    server_model.sequences = weights
    return server_model.get_weights()