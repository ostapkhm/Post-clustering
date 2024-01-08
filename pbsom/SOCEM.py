from pbsom.Utils import Lattice
from pbsom.SOM import SOM

from sklearn.cluster import KMeans

import numpy as np


class SOCEM(SOM):
    def __init__(self, lattice: Lattice, learning_rate, use_weights, lam, random_state=None):
        super().__init__(lattice, learning_rate, use_weights, random_state)

        self.lambda_ = lam
        

    def fit(self, X: np.ndarray, epochs, monitor=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        neurons_nb = self.lattice_.neurons_nb_
        neurons = self.lattice_.neurons_

        ### Initial estimates using KMeans
        k_means = KMeans(n_clusters=neurons_nb)
        k_means.fit(X)

        for i in range(0, neurons_nb):
            idxs = (k_means.labels_ == i)

            neurons[i].weight_ = np.sum(idxs)/X.shape[0]
            neurons[i].mean_ = k_means.cluster_centers_[i]
            neurons[i].cov_ = np.diag(np.var(X[idxs], axis=0))

        #############################################

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

            clusters = [[] for _ in range(neurons_nb)]
            for x in X:
                # Find responsibilities for each x
                # E step
                responsibilities = self.find_responsibilities(x)
                
                # Define cluster for x based on largest posterior probability
                # C step
                k = np.argmax(responsibilities)
                clusters[k].append(x)

            #############

            # Update nodes
            ### M-step ###
            self.update_nodes(clusters)

    def neighbourhood_func(self, r, s):
        alpha = 0.5 / self.sigma_**2
        return np.exp(-alpha * self.distance(self.lattice_.neurons_[r].coord_, self.lattice_.neurons_[s].coord_))
    
    def neuron_activation(self, x, neuron_idx):
        neuron = self.lattice_.neurons_[neuron_idx]
        d = x.shape[0]
        return np.exp(-0.5*((x - neuron.mean_).T @ np.linalg.inv(neuron.cov_) @ (x - neuron.mean_))) / ((2*np.pi)**(d/2) * np.linalg.det(neuron.cov_)**0.5)

    
    def find_responsibilities(self, x):
        neurons = self.lattice_.neurons_

        # returns responsibilities corresponding to x
        responsibilities = np.zeros(self.lattice_.neurons_nb_)
        
        for k in range(self.lattice_.neurons_nb_):
            enumerator = 0
            denominator = 0

            for l in range(self.lattice_.neurons_nb_):
                enumerator += self.neighbourhood_func(k, l) * np.log(self.neuron_activation(x, l))
            
            for j in range(self.lattice_.neurons_nb_):
                val = 0
                for l in range(self.lattice_.neurons_nb_):
                    val += self.neighbourhood_func(j, l) * np.log(self.neuron_activation(x, l))

                if self.use_weights_:
                    denominator += np.exp(val) * neurons[k].weight_
                else:
                    denominator += np.exp(val)

            responsibilities[k] = np.exp(enumerator) / denominator

            if self.use_weights_:
                responsibilities[k] *= neurons[k].weight_

        return responsibilities


    @staticmethod
    def distance(v1: np.ndarray, v2: np.ndarray):
        # Calculating euclidian distance
        return np.sqrt(np.sum((v1 - v2) ** 2))

    def update_nodes(self, clusters):
        # Updating all neuron based on batch mode

        weight_denom = 0
        for neuron_idx in range(self.lattice_.neurons_nb_):
            weight_denom += len(clusters[neuron_idx])
        
        old_mean = 0
        for neuron_idx in range(self.lattice_.neurons_nb_):
            old_mean += self.lattice_.neurons_[neuron_idx].mean_

        for neuron_idx in range(self.lattice_.neurons_nb_):
            numerator_mean = 0
            numerator_cov = 0
            denominator = 0
            
            
            for k in range(self.lattice_.neurons_nb_):
                val = self.neighbourhood_func(k, neuron_idx)
                for x in clusters[k]:
                    numerator_mean += val * x
                
                denominator += val * len(clusters[k])

            ########
            # Upgrade
            res = np.linalg.inv(np.eye(N=self.lattice_.neurons_[neuron_idx].cov_.shape[0]) * denominator - 2*self.lambda_*(self.lattice_.neurons_nb_ - 1) * self.lattice_.neurons_[neuron_idx].cov_) @ \
                    (numerator_mean - 2*self.lambda_*self.lattice_.neurons_[neuron_idx].cov_ @ (old_mean - self.lattice_.neurons_[neuron_idx].mean_))
            
            ########

            
            self.lattice_.neurons_[neuron_idx].mean_ = res
            mean = self.lattice_.neurons_[neuron_idx].mean_

            for k in range(self.lattice_.neurons_nb_):
                val = self.neighbourhood_func(k, neuron_idx)
                for x in clusters[k]:
                    numerator_cov += val * np.outer(x - mean, x - mean)
            
            self.lattice_.neurons_[neuron_idx].cov_ = numerator_cov / denominator
            self.lattice_.neurons_[neuron_idx].weight_ = len(clusters[neuron_idx]) / weight_denom

        
    def predict(self, X):
        probabilities = np.zeros((X.shape[0], self.lattice_.neurons_nb_))

        for i in range(X.shape[0]):
            # Find best matching unit for x_n
            winner_idx = self.find_bmu(X[i])
            for neuron_idx in range(self.lattice_.neurons_nb_):
                probabilities[i, neuron_idx] = self.neighbourhood_func(winner_idx, neuron_idx)
        
        return np.argmax(probabilities, axis=1)