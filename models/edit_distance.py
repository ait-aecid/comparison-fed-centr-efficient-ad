from models._thresholds import apply_threshold, supervised_threshold_selection
from models._imodel import Model

from typing import Any, List, Tuple
import Levenshtein as Leve
from tqdm import tqdm 


def levenshtein_distance(
    seq_1: Tuple[int], seq_2: Tuple[int], score_cutoff: float | None = None
) -> float | int:
    return Leve.distance(s1=seq_1, s2=seq_2, score_cutoff=score_cutoff)


class EditDistance(Model):
    def __init__(self) -> None:
        self.sequences = set()

    def __add__(self, value: List[Any]) -> None:
        self.sequences.add(tuple(value))

    def set_weights(self, weights: List[Any]) -> None:
        self.sequences = set([eval(seq) for seq in weights])

    def get_weights(self) -> List[Any]:
        return [str(seq) for seq in self.sequences]
    
    def set_threshold(
        self, X_normal: List[List[Any]], X_abnormal: List[List[Any]]
    ) -> None:
        self.threshold = supervised_threshold_selection(
            score_abnormal=self.score(X_abnormal), score_normal=self.score(X_normal)
        )

    def score(self, X: List[List[Any]]) -> List[int]:
        if len(self.sequences) == 0:
            return [-1 for _ in X]
        scores, cache = [], {}
        for xi in tqdm(X):
            if (xi_ := tuple(xi)) in cache.keys():
                scores.append(cache[xi_])
            else:
                scores.append(min([
                    levenshtein_distance(xi_, seq) for seq in self.sequences
                ]))
                cache[xi_] = scores[-1]
        return scores  # TODO: ADD score_cutoff

    def fit(self, X: List[List[Any]]) -> float:
        for xi in X:
            self + xi
        return 0.
    
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