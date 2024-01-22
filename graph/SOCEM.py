from graph.Utils import Lattice
from pbsom.SOM import SOM

from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.linalg import inv
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
            self.H_ = np.exp(-0.5 * self.lattice_.pairwise_distance_ / self.sigma_**2)

            if monitor is not None:
                monitor.save()

            ### E and C-step ###
            clusters = self.predict(X)

            ### M-step ###
            self.update_nodes(X, clusters)

            convergence = np.abs(self.log_likelihood - prev_log_likelihood)
            prev_log_likelihood = self.log_likelihood
            iter_nb += 1
        
        # print("Link cutting started")
        # ### Edge cutting ###
        # self.edge_cutting(X)

    def neuron_activation(self, X):
        # Return value of neuron activation function to X on every neuron
        activation_vals = np.zeros((X.shape[0], self.lattice_.neurons_nb_))

        for i, neuron in enumerate(self.lattice_.neurons_.values()):
            activation_vals[:, i] = multivariate_normal.pdf(X, neuron.mean_, neuron.cov_)

        # delta_min for numerical issues
        delta_min = 2.225e-308

        activation_vals[activation_vals < delta_min] = delta_min
        return activation_vals
    
    
    def find_responsibilities(self, X):
        # E-step

        # For numerical issues
        nu = 744

        neurons = self.lattice_.neurons_
        neurons_nb = self.lattice_.neurons_nb_
        neurons_indexes = list(neurons.keys())

        log_neuron_activation = np.log(self.neuron_activation(X))
        
        responsibilities = log_neuron_activation @ self.H_
        log_likelihoods = responsibilities

        # Corner case
        max_vals = np.max(responsibilities, axis=1)
        mask = max_vals < -nu
        responsibilities[mask] -= max_vals[mask, np.newaxis]
        
        responsibilities = np.exp(responsibilities)

        if self.use_weights_:
            for k in range(neurons_nb):
                responsibilities[:, k] *= neurons[neurons_indexes[k]].weight_
                log_likelihoods[:, k] += np.log(neurons[neurons_indexes[k]].weight_)

        return responsibilities / np.sum(responsibilities), log_likelihoods


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

        # E step
        responsibilities, partial_ll = self.find_responsibilities(X)

        # C step
        clusters_indexes = np.argmax(responsibilities, axis=1)

        for i, k in enumerate(clusters_indexes):
            if return_ll_matrix:
                avg_ll_matrix[k] += partial_ll[i] - np.log(neurons[neurons_indexes[k]].weight_)
                occurrences[k] += 1

            y_pred[i] = neurons_indexes[k]
            log_likelihood += partial_ll[i, k]
        
        # Save log_likelihood
        self.log_likelihood = log_likelihood

        if return_ll_matrix:
            avg_ll_matrix /= occurrences.reshape(-1, 1)
            return y_pred, avg_ll_matrix
        
        return y_pred

    def update_nodes(self, X, clusters):
        # M-step
        # Updating all neuron based on batch mode

        # For numerical issues
        reg_covar = self.reg_covar_ * np.eye(self.n_features_in_)

        cluster_sums = np.zeros((self.lattice_.neurons_nb_, X.shape[1]))
        cluster_counts = np.zeros(self.lattice_.neurons_nb_)
        quadratic_form = np.zeros((self.lattice_.neurons_nb_, X.shape[1], X.shape[1]))

        for i in range(X.shape[0]):
            quadratic_form[clusters[i]] += np.outer(X[i], X[i])
            cluster_sums[clusters[i]] += X[i]
            cluster_counts[clusters[i]] += 1

        denominator = (self.H_ @ cluster_counts)
        mean = (self.H_ @ cluster_sums) / denominator[:, np.newaxis]

        for i, neuron_idx in enumerate(self.lattice_.neurons_):
            self.lattice_.neurons_[neuron_idx].mean_ = mean[i]
        
            cov = np.outer(mean[i], mean[i]) * (self.H_[i] @ cluster_counts) + \
                    + np.sum(self.H_[i][:, np.newaxis, np.newaxis] * quadratic_form, axis=0) \
                    - np.outer(mean[i], self.H_[i] @ cluster_sums) \
                    - np.outer(self.H_[i] @ cluster_sums, mean[i])

            cov /= denominator[i]
            
            self.lattice_.neurons_[neuron_idx].cov_ = cov + reg_covar
            self.lattice_.neurons_[neuron_idx].weight_ = cluster_counts[i] / len(X)
    
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
