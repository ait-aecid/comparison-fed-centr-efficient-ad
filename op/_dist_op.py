
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

        np.random.seed(self.seed)
        np.random.shuffle(values)
        return (values / sum_values).tolist()


class LineDist(UniformDist):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)

    def _equation(self, x: np.ndarray) -> np.ndarray:
        return - x + 1.2

    def sample(self, n: int) -> np.ndarray:
        return self._equation(np.linspace(start=0, stop=1, num=n)) 


class LogNormalDist(UniformDist):
    def __init__(self, seed: int, mu: float = 0, std: float = 0.25) -> None:
        super().__init__(seed)
        self.mu, self.std = mu, std

    def _equation(self, x: np.ndarray) -> np.ndarray:
        value_1 = 1 / (x * self.std * np.sqrt(2 * np.pi))
        value_2 = - (np.log(x) - self.mu) ** 2 / (2 * self.std ** 2)
        return value_1 * np.exp(value_2)

    def sample(self, n: int) -> np.ndarray:
        return self._equation(np.linspace(start=0.5, stop=2., num=n)) 


_dist_methods = {
    "Uniform": UniformDist, 
    "Line": LineDist,
    "LogNormal": LogNormalDist,
}


def split_by_dist(args: Arguments, n_idxs: int) -> t.Iterator[t.Tuple[int, int]]:
    dist = _dist_methods[args.dist_method](args.seed_number)(args.amount_clients)
    value = 0
    for i in range(args.amount_clients):
        old_value = value
        value += round(n_idxs * dist[i])
        yield (old_value, value)