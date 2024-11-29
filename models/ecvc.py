"""
ECVC method from: https://dl.acm.org/doi/pdf/10.1145/3660768

Pseudo code:
------------
score = min(manh(x, xi) / max(x, xi) for xi in X_train)
is_abnormal = score >= threshold 

Federated approach: 
-------------------
for weigth in client_weights:
    server_weights.extend(weight)
"""
from models._thresholds import supervised_threshold_selection, apply_threshold
from models._imodel import Model

from typing import Any, List, Tuple
from tqdm import tqdm
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


def max_elemnt_wise(X1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    X2 = torch.broadcast_to(x2, X1.shape)

    new_x = torch.cat((X1.reshape((1, -1)), X2.reshape((1, -1))), dim=0).T
    new_x, _ = torch.max(new_x, dim=1)

    return new_x.reshape(X1.shape)


def get_max_elem(X: List[List[int]]) -> int:
    max_ = 0
    for xi in X:
        max_ = max(max_, max(xi))
    return max_


def norm_tensor(x: torch.Tensor) -> torch.Tensor:
    return x / (x.sum(dim=1).view((-1, 1)) + 1e-10)


class ECVC(Model):
    def __init__(self) -> None:
        self.vectors = torch.Tensor([])
        self.n_elemts = 0
        self.is_trained = False

    def __get_vectors(self, X: List[List[Any]]) -> Tuple[torch.Tensor, int]:
        n_elemts = get_max_elem(X) + 1
        vectors = generate_vector(X, n_elemts=n_elemts)
        return vectors, n_elemts

    def get_weights(self) -> List[Any]:
        return self.vectors.detach().tolist()
    
    def set_weights(self, weights: List[Any]) -> None:
        self.vectors = torch.unique(torch.Tensor(weights), dim=0, sorted=False)

    def fit(self, X: List[List[Any]]) -> float:
        vectors, self.n_elemts = self.__get_vectors(X)
        self.set_weights(vectors)
        self.is_trained = True

        return len(self.vectors)

    def set_threshold(
        self, X_normal: List[List[Any]], X_abnormal: List[List[Any]]
    ) -> None:
        self.threshold = supervised_threshold_selection(
            score_normal=self.score(X_normal),
            score_abnormal=self.score(X_abnormal),
        )
        print(self.threshold)
    
    def score(self, X: List[List[Any]], batch_size: int = 2000) -> List[float]:
        if not self.is_trained:
            return torch.zeros(len(X)).detach().tolist()
        
        min_dist = []
        for i in tqdm(range(0, len(X), batch_size)):
            train_vectors = self.vectors
            vectors, n_elemts = self.__get_vectors(X[i:i + batch_size])
            if self.n_elemts != n_elemts:
                train_vectors, vectors = convert_same_shape(train_vectors, vectors)

            norm_train, norm_vecs = norm_tensor(train_vectors), norm_tensor(vectors)
            for norm_vec in norm_vecs:
                manh = torch.abs(norm_train - norm_vec).sum(dim=1)
                limit = max_elemnt_wise(X1=train_vectors, x2=norm_vec).sum(dim=1)
                dist, _ = torch.min(manh / limit, dim=0)
                min_dist.append(dist.detach().tolist())

        return min_dist  # TODO:  add idf weights

    def predict(self, X: List[List[Any]]) -> List[int]:
        return apply_threshold(self.score(X), threshold=self.threshold)


def update_strategy(
    server_model: ECVC, clients_weights: List[List[Any]]
) -> List[Any]:

    max_len_w = torch.Tensor([[]])
    for weights in clients_weights:
        if  weights != [] and len(weights[0]) > max_len_w.shape[1]:
            max_len_w = torch.Tensor(weights)

    conver_weights = []
    for weights in clients_weights:
        if weights != []:
            x1, _ = convert_same_shape(
                x1=torch.Tensor(weights), x2=max_len_w
            )
            conver_weights.extend(x1.detach().tolist())

    server_model.set_weights(conver_weights) 
    server_model.is_trained = True
    return server_model.get_weights()