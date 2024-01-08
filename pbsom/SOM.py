from abc import ABC, abstractmethod

class SOM(ABC):
    def __init__(self, lattice, learning_rate, use_weights=False, random_state=None):
        self.learning_rate_ = learning_rate
        self.sigma_ = None
        self.lattice_ = lattice
        self.random_state = random_state
        self.use_weights_ = use_weights
    
    @abstractmethod
    def fit(self, X, epochs, monitor=None):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @staticmethod
    @abstractmethod
    def distance(v1, v2):
        pass