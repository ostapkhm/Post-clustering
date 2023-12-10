from abc import ABC, abstractmethod
import numpy as np

class Neuron:
    def __init__(self, weight, mean, cov, coord):
        self.weight_ = weight
        self.mean_ = mean
        self.cov_ = cov
        self.coord_ = coord
    
class Lattice(ABC):
    def __init__(self, size):
        self.size_ = size
        self.neurons_ = None
        self.neurons_nb_ = None

        self.generate()

    @abstractmethod
    def generate(self):
        pass


class RectangularLattice(Lattice):
    def generate(self):
        self.neurons_ = []
        self.neurons_nb_ = self.size_[0] * self.size_[1]

        for i in range(0, self.size_[0]):
            for j in range(0, self.size_[1]):
                self.neurons_.append(Neuron(None, None, None, np.array([i, j])))
