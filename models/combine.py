
from models._imodel import Model

import numpy as np
import typing as t


class Combine(Model):
    def __init__(
        self, 
        models: t.List[Model],
        update_funcs: t.Callable[[Model, t.List[t.List[t.Any]]], None] | None = None
    ) -> None:
        super().__init__()
        self.models = models
        self.update_funcs = update_funcs 

    def set_weights(self, weights: t.List[t.Any]) -> None:
        for model, weights_ in zip(self.models, weights):
            model.set_weights(eval(weights_))

    def get_weights(self) -> t.List[t.Any]:
        weights = []
        for model in self.models:
            weights.append(str(model.get_weights()))
        return weights

    def set_threshold(
        self, X_normal: t.List[t.List[t.Any]], X_abnormal: t.List[t.List[t.Any]]
    ) -> None:
        for model in self.models:
            model.set_threshold(X_normal=X_normal, X_abnormal=X_abnormal)

    def fit(self, X: t.List[t.List[t.Any]]) -> float:
        for model in self.models:
            model.fit(X)
        return 0.0

    def predict(self, X: t.List[t.List[t.Any]]) -> t.List[int]:
        results = self.models[0].predict(X)
        for model in self.models[1:]:
            results = np.logical_or(results, model.predict(X))
        return results.astype(int).tolist()


def update_strategy(
    server_model: Combine, clients_weights: t.List[t.List[t.Any]]
) -> t.List[t.Any]:
    for j, update in enumerate(server_model.update_funcs):
        update(
            server_model=server_model.models[j], 
            clients_weights=[eval(w[j]) for w in clients_weights]
        )
    return server_model.get_weights()