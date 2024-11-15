
from abc import ABC, abstractmethod

import typing as t


class Model(ABC):
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
    def fit(X: t.List[t.List[t.Any]]) -> float: 
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