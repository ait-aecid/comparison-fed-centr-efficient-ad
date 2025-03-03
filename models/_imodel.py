"""
Main class to be use for all the methods/models in the repository
"""
from abc import ABC, abstractmethod

import typing as t


class Model(ABC):
    def __init__(self, name: str = "Model", thres: float | None = None) -> None:
        self.name = name
        self.threshold = thres

    @abstractmethod
    def set_weights(self, weights: t.List[t.Any]) -> None: 
        """
        Set the weights/parameters of the model
        """
        pass

    @abstractmethod
    def get_weights(self) -> t.List[t.Any]: 
        """ 
        Set the weights/parameters of the model
        """
        pass

    @abstractmethod
    def fit(self, X: t.List[t.List[t.Any]]) -> float: 
        """
        Method to train the model
        """
        pass

    @abstractmethod
    def predict(self, X: t.List[t.List[t.Any]]) -> t.List[int]:
        """
        Method to do the prediction 
        """
        pass

    def set_threshold(
        self, X_normal: t.List[t.List[t.Any]], X_abnormal: t.List[t.List[t.Any]]
    ) -> None:
        """
        Method to setup the threshold (model must be trained first)
        """
        pass