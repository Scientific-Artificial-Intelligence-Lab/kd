from abc import ABCMeta, abstractmethod
from ..base import BaseEstimator





class BaseGa(BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for genetic algorithm based models."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        """Fit model to data."""

    def predict(self, X):
        """Make predictions."""
        pass



class BaseRL(BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for reinforcement learning based models."""
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        """Fit model to data."""

    def predict(self, X):
        """Make predictions."""
        pass