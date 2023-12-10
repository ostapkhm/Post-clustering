from Utils import Lattice
from SOM import SOM

from scipy.stats import multivariate_normal
import numpy as np


class SOEM(SOM):
    def __init__(self, lattice: Lattice, learning_rate, random_state=None):
        super().__init__(lattice, learning_rate, random_state)

    def fit(self, X: np.ndarray, epochs, monitor=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        neurons_nb = self.lattice_.neurons_nb_
        neurons = self.lattice_.neurons_


        for i in range(0, neurons_nb):
            neurons[i].weight_ = 1 / neurons_nb
            neurons[i].mean_ = self.initialize_mean(X)
            neurons[i].cov_ = self.initialize_cov(X)

        for ep in range(epochs):
            self.sigma_ = 1/(np.sqrt(ep + 1) * self.learning_rate_)
            print("Epoch:", ep)
            print("sigma^2:", self.sigma_**2)
            print("Weights, Covariance:")
            for neuron in self.lattice_.neurons_:
                print(neuron.mean_, end=' ')
                print()
                print(neuron.cov_)
                print()
            print("-----------------")

            if monitor is not None:
                monitor.save()
            
            
            responsibilities = np.zeros(shape=(len(X), neurons_nb))
            for t, x in enumerate(X):
                # Find responsibilities for each x
                # E step
                responsibilities[t] = self.find_responsibilities(x)

            print("Responsobilities->", np.isnan(responsibilities).any())

            for neuron_idx in range(self.lattice_.neurons_nb_):
                # Update nodes
                ### M-step ###

                numerator_mean = 0
                denominator = 0
                numerator_cov = 0

                for i, x in enumerate(X):
                    val = 0
                    for k in range(self.lattice_.neurons_nb_):
                        val += self.neighbourhood_func(k, neuron_idx) * responsibilities[i, k]
                    
                    numerator_mean += val * x
                    denominator += val

                self.lattice_.neurons_[neuron_idx].mean_ = numerator_mean / denominator
                mean = self.lattice_.neurons_[neuron_idx].mean_

                for i, x in enumerate(X):
                    val = 0
                    for k in range(self.lattice_.neurons_nb_):
                        val += self.neighbourhood_func(k, neuron_idx) * responsibilities[i, k]

                    numerator_cov += val * np.outer(x - mean, x - mean)
                    
                self.lattice_.neurons_[neuron_idx].cov_ = numerator_cov / denominator
                self.lattice_.neurons_[neuron_idx].weight_ = np.sum(responsibilities[:, neuron_idx]) / len(X)


    def neighbourhood_func(self, r, s):
        alpha = 0.5 / self.sigma_**2
        return np.exp(-alpha * self.distance(self.lattice_.neurons_[r].coord_, self.lattice_.neurons_[s].coord_))
    
    def neuron_activation(self, x, neuron_idx):
        neuron = self.lattice_.neurons_[neuron_idx]
        res = multivariate_normal.pdf(x=x, mean=neuron.mean_, cov=neuron.cov_)
        
        # numerical issues
        delta_min = 2.2251 * 10**(-308)
        if res < delta_min:
            res = delta_min

        return res

    
    def find_responsibilities(self, x):
        # Parameter for numerical issue
        nu = 450

        # returns responsibilities corresponding to x
        responsibilities = np.zeros(self.lattice_.neurons_nb_)

        denominator = 0
        for j in range(self.lattice_.neurons_nb_):
            val = 0
            for l in range(self.lattice_.neurons_nb_):
                val += self.neighbourhood_func(j, l) * np.log(self.neuron_activation(x, l))

            denominator += np.exp(val + nu)

        
        for k in range(self.lattice_.neurons_nb_):
            enumerator = 0

            for l in range(self.lattice_.neurons_nb_):
                enumerator += self.neighbourhood_func(k, l) * np.log(self.neuron_activation(x, l))
                
            responsibilities[k] = np.exp(enumerator + nu) / denominator
        
        # print("Denom->", denominator)

        return responsibilities


    def predict(self, X):
        probabilities = np.zeros((X.shape[0], self.lattice_.neurons_nb_))

        for i in range(X.shape[0]):
            # Find best matching unit for x_n
            winner_idx = self.find_bmu(X[i])
            for neuron_idx in range(self.lattice_.neurons_nb_):
                probabilities[i, neuron_idx] = self.neighbourhood_func(winner_idx, neuron_idx)
        
        return np.argmax(probabilities, axis=1)

    @staticmethod
    def distance(v1: np.ndarray, v2: np.ndarray):
        # Calculating euclidian distance
        return np.sqrt(np.sum((v1 - v2) ** 2))
        

    @staticmethod
    def initialize_mean(X):
        mean = np.zeros(X.shape[1])
        for i in range(0, X.shape[1]):
            mean[i] = np.random.uniform(np.min(X[:, i]), np.max(X[:, i]))

        return mean
    

    @staticmethod
    def initialize_cov(X):
        cov_matrix = np.zeros((X.shape[1], X.shape[1]))
        for i in range(0, X.shape[1]):
            cov_matrix[i, i] = np.random.uniform(0, (np.max(X[:, i]) - np.min(X[:, i]))/6)**2
            
        return cov_matrix