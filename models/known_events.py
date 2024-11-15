import typing as t


class KnownEvents:
    def __init__(self) -> None:
        self.events = set()

    def __contains__(self, x: t.Any) -> bool:
        return x in self.events

    def __add__(self, x: t.Any) -> t.Self:
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

