from abc import ABC, abstractmethod

class SOM(ABC):
    def __init__(self, lattice, learning_rate, tol=1e-4, max_iter=100, use_weights=False, random_state=None):
        self.sigma_ = None
        self.log_likelihood = None
        self.lattice_ = lattice
        self.learning_rate_ = learning_rate
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.use_weights_ = use_weights
        self.random_state = random_state
    

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