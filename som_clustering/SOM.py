from abc import ABC, abstractmethod

class SOM(ABC):
    def __init__(self, lattice, sigma_start, sigma_step, tol=1e-4, max_iter=100, use_weights=False, random_state=None, reg_covar=1e-6):
        self.sigma_ = sigma_start
        self.log_likelihood_ = None
        self.lattice_ = lattice
        
        self.sigma_start_ = sigma_start
        self.sigma_step_ = sigma_step
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.use_weights_ = use_weights
        self.random_state = random_state
        
        # H matrix - lateral interaction between neurons
        self.H_ = None
        self.n_features_in_ = None
        self.reg_covar_ = reg_covar
        

    @abstractmethod
    def fit(self, X, epochs, monitor=None):
        pass
    

    @abstractmethod
    def predict(self, X):
        pass