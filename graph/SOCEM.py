from graph.Utils import Lattice
from pbsom.SOM import SOM

from sklearn.cluster import KMeans
import numpy as np


class SOCEM(SOM):
    def __init__(self, lattice: Lattice, learning_rate, betta, tol=1e-4, max_iter=100, use_weights=False, random_state=None, reg_covar=1e-6):
        super().__init__(lattice, learning_rate, tol, max_iter, use_weights, random_state, reg_covar)
        self.betta_ = betta
        

    def fit(self, X: np.ndarray, monitor=None):
        self.n_features_in_ = X.shape[1]
        neurons_nb = self.lattice_.neurons_nb_
        neurons = self.lattice_.neurons_

        if monitor is not None:
            monitor.initialize_params()

        ### Initial estimates using KMeans ###
        k_means = KMeans(n_clusters=neurons_nb, random_state=self.random_state)
        k_means.fit(X)

        # For numerical issues
        reg_covar = self.reg_covar_ * np.eye(self.n_features_in_)
        for idx in neurons.keys():
            idxs = (k_means.labels_ == idx)

            neurons[idx].weight_ = np.sum(idxs) / X.shape[0]
            neurons[idx].mean_ = k_means.cluster_centers_[idx]
            neurons[idx].cov_ = np.diag(np.var(X[idxs], axis=0)) + reg_covar

        #############################################

        iter_nb = 0
        prev_log_likelihood = 0
        convergence = np.inf

        while iter_nb < self.max_iter_ and convergence > self.tol_:
            self.sigma_ = 1 / (np.sqrt(iter_nb + 1) * self.learning_rate_)

            if monitor is not None:
                monitor.save()

            # E and C step
            clusters = self.predict(X)
            ### M-step ###
            self.update_nodes(X, clusters)

            convergence = np.abs(self.log_likelihood - prev_log_likelihood)
            prev_log_likelihood = self.log_likelihood
            iter_nb += 1
        
        # print("Link cutting started")
        # ### Edge cutting ###
        # self.edge_cutting(X)
        
    
    def edge_cutting(self, X):
        neurons_nb = self.lattice_.neurons_nb_
        neurons_keys = list(self.lattice_.neurons_.keys())
        to_idx_map = {key: index for index, key in enumerate(neurons_keys)}

        adj_list = self.lattice_.graph_.adj_list_

        _, avg_ll_matrix = self.predict(X, return_ll_matrix=True)
        h = np.max(-np.diag(avg_ll_matrix))

        divergence_matrix = np.zeros((neurons_nb, neurons_nb))
        for k in range(neurons_nb):
            divergence_matrix[k] = avg_ll_matrix[k, k] - avg_ll_matrix[k]
    

        for e in adj_list.keys():
            for v in adj_list[e]:
                idx_e = to_idx_map[e]
                idx_v = to_idx_map[v]

                if 0.5 * (divergence_matrix[idx_e, idx_v] + divergence_matrix[idx_v, idx_e]) > self.betta_ * h:
                    print(e, v)
                    self.lattice_.delete_edge(e, v)
        
        self.lattice_.update_distances()

    def neighbourhood_func(self, r, s):
        alpha = 0.5 / self.sigma_**2
        return np.exp(-alpha * self.lattice_.get_pairwise_distance(r, s))
    

    def neuron_activation(self, x, neuron_idx):
        neuron = self.lattice_.neurons_[neuron_idx]
        d = x.shape[0]
        exponential_term = np.exp(-0.5 * ((x - neuron.mean_).T @ np.linalg.inv(neuron.cov_) @ (x - neuron.mean_)))
        normalization_factor = ((2 * np.pi) ** (d / 2)) * np.linalg.det(neuron.cov_) ** 0.5

        # delta_min for numerical issues
        delta_min = 2.225e-308

        res = exponential_term / normalization_factor
        if res < delta_min:
            return delta_min

        return res
    
    
    def find_responsibilities(self, x):
        # returns responsibilities corresponding to x

        # for numerical issues
        nu = 744

        neurons = self.lattice_.neurons_
        neurons_nb = self.lattice_.neurons_nb_
        neurons_indexes = list(neurons.keys())

        responsibilities = np.zeros(neurons_nb)
        log_neuron_activation = np.zeros(neurons_nb)

        for l in range(neurons_nb):
            log_neuron_activation[l] = np.log(self.neuron_activation(x, neurons_indexes[l]))
        
        for k in range(neurons_nb):
            enumerator = 0
            for l in range(neurons_nb):
                enumerator += self.neighbourhood_func(neurons_indexes[k], neurons_indexes[l]) * log_neuron_activation[l]
            
            responsibilities[k] = enumerator
        
        log_likelihoods = responsibilities
        # corner case
        if np.max(responsibilities) < -nu:
            responsibilities -= np.max(responsibilities)

        responsibilities = np.exp(responsibilities)

        if self.use_weights_:
            for k in range(neurons_nb):
                responsibilities[k] *= neurons[neurons_indexes[k]].weight_

        return responsibilities / np.sum(responsibilities), log_likelihoods

    def update_nodes(self, X, clusters):
        # Updating all neuron based on batch mode
        neurons_indexes = list(self.lattice_.neurons_.keys())

        # for numerical issues
        reg_covar = self.reg_covar_ * np.eye(self.n_features_in_)

        for neuron_idx in neurons_indexes:
            numerator_mean = 0
            denominator = 0
            
            for k in neurons_indexes:
                val = self.neighbourhood_func(k, neuron_idx)

                numerator_mean += val * np.sum(X[clusters==k], axis=0)
                denominator += val * len(X[clusters==k])
            
            self.lattice_.neurons_[neuron_idx].mean_ = numerator_mean / denominator
            mean = self.lattice_.neurons_[neuron_idx].mean_
            
            numerator_cov = 0
            for k in range(self.lattice_.neurons_nb_):
                val = self.neighbourhood_func(k, neuron_idx)
                for x in X[clusters==k]:
                    numerator_cov += val * np.outer(x - mean, x - mean)
            
            self.lattice_.neurons_[neuron_idx].cov_ = numerator_cov / denominator + reg_covar
            self.lattice_.neurons_[neuron_idx].weight_ = len(X[clusters==neuron_idx]) / len(X)


    def predict(self, X, return_ll_matrix=False):
        neurons = self.lattice_.neurons_
        neurons_indexes = list(neurons.keys())
        neurons_nb = self.lattice_.neurons_nb_
        avg_ll_matrix = None
        occurrences = None

        if return_ll_matrix:
            avg_ll_matrix = np.zeros((neurons_nb, neurons_nb))
            occurrences = np.zeros(neurons_nb)


        y_pred = np.zeros(X.shape[0], dtype='int32')
        log_likelihood = 0

        for i, x in enumerate(X):
            # Find responsibilities for each x
            # E step
            responsibilities, partial_ll = self.find_responsibilities(x)

            # Define cluster for x based on largest posterior probability
            # C step
            k = np.argmax(responsibilities)

            if return_ll_matrix:
                avg_ll_matrix[k] += partial_ll
                occurrences[k] += 1

            y_pred[i] = neurons_indexes[k]
            log_likelihood += partial_ll[k]
        
        # Save log_likelihood
        self.log_likelihood = log_likelihood

        if return_ll_matrix:
            for k in range(neurons_nb):
                avg_ll_matrix[k] /= occurrences[k]

            return y_pred, avg_ll_matrix
        
        return y_pred
    