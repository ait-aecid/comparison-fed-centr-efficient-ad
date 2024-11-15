
from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def set_weights(self, weights): pass

    @abstractmethod
    def get_weights(self, weights): pass

    @abstractmethod
    def fit(X): pass

    @abstractmethod
    def predict(self, X): pass