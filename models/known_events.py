
from models._imodel import Model

import typing as t


class KnownEvents(Model):
    def __init__(self) -> None:
        self.events = set()

    def __contains__(self, x: t.Any) -> bool:
        return x in self.events

    def __add__(self, x: t.Union[t.Any, t.Set[t.Any]]) -> t.Self:
        if isinstance(x, set):
            self.set_weights(self.events.union(x))
        else:
            self.events.add(x)
        return self

    def __len__(self) -> int:
        return len(self.events)

    def set_weights(self, events: t.Set[t.Any])-> None:
        self.events = events

    def get_weights(self) -> t.Set[t.Any]:
        return self.events

    def fit(self, X: t.List[t.List[t.Any]]) -> int:
        for xi in X:
            for event in set(xi):
                self += event
        return len(self)

    def predict(self, X: t.List[t.List[t.Any]]) -> t.List[int]:
        results = []
        for xi in X:
            results.append(0)
            for event in set(xi):
                if event not in self:
                    results[-1] = 1
                    break 
        return results


def update_strategy(
    server_model: KnownEvents, clients_weights: t.List[t.Set[t.Any]]
) -> t.Set[t.Any]:
    for client_weights in clients_weights:
        server_model += client_weights
    return server_model.get_weights()


