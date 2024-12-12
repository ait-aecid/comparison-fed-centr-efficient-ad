
import typing as t
import numpy as np


Arguments = t.NewType("Arguments", object)


class UniformDist:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def sample(self, n: int) -> np.ndarray:
        return np.ones(n)

    def __call__(
        self, n_clients: int, plot_path: str | None = None
    ) -> t.List[float]:
        """
        The output is the sample distribution (dist) where sum(dist) = 1
        """
        values = self.sample(n_clients)
        sum_values = np.sum(values)
        return (values / sum_values).tolist()


def split_by_dist(args: Arguments, n_idxs: int) -> t.Iterator[t.Tuple[int, int]]:
    dist = UniformDist(args.seed_number)(args.amount_clients)
    value = 0
    for i in range(args.amount_clients):
        old_value = value
        value += int(n_idxs * dist[i])
        yield (old_value, value)