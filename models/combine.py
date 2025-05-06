"""
Combine methods from: https://dl.acm.org/doi/pdf/10.1145/3660768

Pseudo code:
------------
Methods -> [method_1, method_2 ..]
X -> [(sequence, next_event) ... ]
prediction = method_1(X) or method_2(X) ... 

Federated approach: 
-------------------
use the update strategy of each method individually
"""
from models._imodel import Model

import numpy as np

from typing import List
import typing as t


class Combine(Model):
    """
    Combine methods, it use the union of the results of each method.
    """
    def __init__(
        self, 
        models: List[Model],
        update_funcs: List[t.Callable[[Model, List[List[t.Any]]], None]] | None = None
    ) -> None:
        super().__init__(name=" ".join(m.name for m in models))
        self.models = models
        self.update_funcs = update_funcs 

    def set_weights(self, weights: List[t.Any]) -> None:
        for model, weights_ in zip(self.models, weights):
            model.set_weights(eval(weights_))

    def get_weights(self) -> List[t.Any]:
        weights = []
        for model in self.models:
            weights.append(str(model.get_weights()))
        return weights

    def set_threshold(
        self, X_normal: List[List[t.Any]], X_abnormal: List[List[t.Any]]
    ) -> None:
        for model in self.models:
            model.set_threshold(X_normal=X_normal, X_abnormal=X_abnormal)

    def fit(self, X: List[List[t.Any]]) -> float:
        for model in self.models:
            model.fit(X)
        return 0.0

    def predict(self, X: List[List[t.Any]]) -> t.List[int]:
        results = self.models[0].predict(X)
        for model in self.models[1:]:
            results = np.logical_or(results, model.predict(X))
        return results.astype(int).tolist()


def update_strategy(
    server_model: Combine, clients_weights: List[List[t.Any]]
) -> List[t.Any]:
    for j, update in enumerate(server_model.update_funcs):
        update(
            server_model=server_model.models[j], 
            clients_weights=[eval(w[j]) for w in clients_weights]
        )
    return server_model.get_weights()