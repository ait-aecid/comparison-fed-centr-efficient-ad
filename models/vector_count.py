
from models._imodel import Model

from typing import Any, List, Tuple
import torch


def generate_vector(x: List[List[Any]], n_elemts: int) -> torch.Tensor:
    vector = None
    for xi in x:
        x_ = torch.Tensor([xi]).long()
        x_ = torch.nn.functional.one_hot(x_, num_classes=n_elemts).sum(dim=1)
        vector = x_ if vector is None else  torch.cat((vector, x_), dim=0)
    
    return vector


def convert_same_shape(
    x1: torch.Tensor, x2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if (n1 := x1.shape[1]) > (n2 := x2.shape[1]):
        x2 = torch.cat((x2, torch.zeros(x2.shape[0], n1 - n2)), dim=1)
    else:
        x1 = torch.cat((x1, torch.zeros(x1.shape[0], n2 - n1)), dim=1)

    return x1, x2


def get_max_elem(X: List[List[int]]) -> int:
    max_ = 0
    for xi in X:
        max_ = max(max_, max(xi))
    return max_


class CountVector(Model):
    def __init__(self) -> None:
        self.vectors = torch.Tensor([])
        self.n_elemts = 0

    def __get_vectors(self, X: List[List[Any]]) -> Tuple[torch.Tensor, int]:
        n_elemts = get_max_elem(X) + 1
        vectors = torch.unique(generate_vector(X, n_elemts=n_elemts), dim=0)
        return vectors, n_elemts

    def get_weights(self) -> List[Any]:
        return self.vectors.detach().tolist()
    
    def set_weights(self, weights: List[Any]) -> None:
        self.vectors = torch.Tensor(weights)

    def fit(self, X: List[List[Any]]) -> float:
        self.vectors, self.n_elemts = self.__get_vectors(X)
        self.vectors = torch.unique(self.vectors, dim=0)

        return len(self.vectors)
    
    def predict(self, X: List[List[Any]]) -> List[int]:
        def norm_tensor(x: torch.Tensor) -> torch.Tensor:
            return x / (x.sum(dim=0) + 1e-10)

        vectors, n_elemts = self.__get_vectors(X)
        
        train_vectors = self.vectors
        if self.n_elemts != n_elemts:
            train_vectors, vectors = convert_same_shape(train_vectors, vectors)
        norm_train, norm_vecs = norm_tensor(train_vectors), norm_tensor(vectors)
        train_limit, _ = torch.max(norm_train, dim=0)

        min_dist = []
        for norm_vec in norm_vecs:
            manh = torch.abs(norm_train - norm_vec).sum(dim=1)
            limit = torch.max(train_limit, norm_vec).sum()
            dist, _ = torch.min(manh / limit, dim=0)
            min_dist.append(dist)

        return min_dist  # TPDO: add threshold
