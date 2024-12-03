"""
NGRAM method from: https://dl.acm.org/doi/pdf/10.1145/3660768

Pseudo code:
------------
X -> [(sequence, next_event) ... ]
score = max(ngram[sequence]) == next_event
is_anomaly = if score >= threshold

Federated approach: 
-------------------
Join the ngram tables of each client
"""
from models._thresholds import supervised_threshold_selection, apply_threshold
from models._imodel import Model

from typing import Any, List, Tuple, Self
from tqdm import tqdm
import numpy as np


def split_seq(X: List[List[Any]], sep: int) -> List[List[Any]]:
    split = []
    for xi in X:
        x_ = [-1 for _ in range(sep)] + xi
        for j in range(0, len(x_) - sep):
            split.append([x_[j: j + sep], x_[j + sep]])

    return split


class Serilize:
    def convert(data: List[Any]) -> List[Any]:
        conversion = []
        for d in data:
            if len(d) > 0 and isinstance(d[0], list):
                for di in d:
                    conversion.extend(di)
            else:
                conversion.extend(d)
            conversion.append("<END>")

        return conversion[:-1]

    def inverse(serialize_data: List[Any]) -> List[Any]:
        data = []
        list_ = []
        for data_ in serialize_data:
            if data_ == "<END>":
                data.append(list_)
                list_ = []
            else:
                list_.append(data_)
        data.append(list_)
        data[-1] = np.reshape(
            data[-1], (len(data[1]), len(data[0]))
        ).astype(int).tolist()

        return data


class GramMatrix:
    def __init__(self) -> None:
        self.storage = {}
        self.vocab = []

    def get_combs(self) -> List[Any]:
        return list(self.storage.keys())

    def get_values(self) -> Tuple[List[Any], List[Any], List[List[int]]]:
        return [self.vocab, self.get_combs(), list(self.storage.values())]
 
    @classmethod
    def from_values(
        cls, values: Tuple[List[Any], List[Any], List[List[int]]]
    ) -> Self:
        
        matrix = cls()
        matrix.vocab = values[0]
        matrix.storage = {k:v for k, v in zip(values[1], values[2])}
        return matrix

    def __getitem__(self, idx: Any) -> List[Any]:  
        if isinstance(idx, tuple):
            if idx[1] not in self.vocab:
                return 0
            return self.storage[idx[0]][self.vocab.index(idx[1])]
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
        return Serilize.convert(self.matrix.get_values())
    
    def set_weights(self, weights: List[Any]) -> None:
        weights = Serilize.inverse(weights)
        self.matrix = self.matrix.from_values(weights) 

    def set_threshold(
        self, X_normal: List[List[Any]], X_abnormal: List[List[Any]]
    ) -> None:
        self.threshold = supervised_threshold_selection(
            score_normal=self.score(X_normal),
            score_abnormal=self.score(X_abnormal),
        )

    def fit(self, X: List[List[Any]]) -> float:
        for [seq, e] in split_seq(X, sep=self.n):
            self.matrix.update(str(seq), str(e))
        return 0.
    
    def score(self, X: List[List[Any]]) -> List[int]:
        results = []
        for xi in tqdm(X):
            errors_count = 0
            for seq, e in split_seq([xi], sep=self.n):
                if str(seq) not in self.matrix:
                    errors_count += 1
                else:
                    if self.matrix[str(seq), str(e)] == 0:
                        errors_count += 1
            results.append(errors_count)
        return results

    def predict(self, X: List[List[Any]]) -> List[int]:
        return apply_threshold(self.score(X), threshold=self.threshold)


class NGram2(NGram):
    def __init__(self) -> None:
        super().__init__(n=2)


class NGram3(NGram):
    def __init__(self) -> None:
        super().__init__(n=3)


def update_strategy(
    server_model: NGram, clients_weights: List[List[Any]]
) -> List[Any]:
    
    for client_weights in clients_weights:
        client_ngram = NGram(server_model.n)
        client_ngram.set_weights(client_weights)
        
        vocab = client_ngram.matrix.vocab
        combs = list(client_ngram.matrix.get_combs())

        for comb in combs:
            for seq, value in zip(vocab, client_ngram.matrix[comb]):
                for _ in range(value):
                    server_model.matrix.update(comb, seq)

    return server_model.get_weights()
                
