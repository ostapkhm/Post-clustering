from Utils import Neuron, Lattice
from SOM import SOM
from Monitor import Monitor

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal



class SOCEM(SOM):
    def __init__(self, lattice: Lattice, sigma_start, sigma_step, betta, cov_type, method, tol=1e-4, max_iter=100, use_weights=False, random_state=None, reg_covar=1e-6):
        super().__init__(lattice, sigma_start, sigma_step, tol, max_iter, use_weights, random_state, reg_covar)
        self.betta_ = betta
        self.merge_method_ = None
        self.cov_type_ = cov_type
        self.criteria_ = None

        if method == 'bhattacharyya':
            self.merge_method_ = self.delete_vertex_bhattacharyya
        elif method == 'ridgeline':
            self.merge_method_ = self.delete_vertex_ridgeline
        elif method == 'mdl':
            self.merge_method_ = self.delete_vertex_mdl
        

    def fit(self, X: np.ndarray):
        map_changed = True

        self.sigma_ = self.sigma_start_

        monitor = Monitor(self)
        self.estimate_params(X, monitor)
        y_pred = self.predict(X)
        monitor.save(y_pred)

        ### Delete empty clusters ###
        self.delete_empty_clusters()
        monitor.save(y_pred, None, True)

        while map_changed:
            print("Neurons->", self.lattice_.neurons_.keys())
            ### Delete unnecessary vertex based on some criteria ###
            map_changed = self.merge_method_()
            self.delete_empty_clusters()
            y_pred = self.predict(X)
            monitor.save(y_pred, None, True)
            print("Map changed -> ", map_changed)
        
        return monitor

    def estimate_params(self, X: np.ndarray, monitor=None):
        self.n_features_in_ = X.shape[1]
        neurons_nb = self.lattice_.neurons_nb_
        neurons = self.lattice_.neurons_

        ###### Initial estimates using KMeans #######
        k_means = KMeans(n_clusters=neurons_nb, random_state=self.random_state, n_init='auto')
        k_means.fit(X)

        # For numerical issues
        reg_covar = self.reg_covar_ * np.eye(self.n_features_in_)

        for i, idx in enumerate(neurons):
            idxs = (k_means.labels_ == i)

            neurons[idx].weight_ = np.sum(idxs) / X.shape[0]
            neurons[idx].mean_ = k_means.cluster_centers_[i]
            neurons[idx].cov_ = np.cov(X[idxs], rowvar=False) + reg_covar
        
        self.H_ = np.exp(-0.5 * self.lattice_.pairwise_distance_ / self.sigma_**2)

        if monitor is not None:
            monitor.save(k_means.predict(X))

        #############################################
        iter_nb = 0

        while self.sigma_ > 0:
            self.H_ = np.exp(-0.5 * self.lattice_.pairwise_distance_ / self.sigma_**2)

            ### E and C-step ###
            clusters = self.predict(X)

            ### M-step ###
            self.update_nodes(X, clusters)

            if monitor is not None:
                monitor.save(self.predict(X))

            iter_nb += 1
            self.sigma_ -= self.sigma_step_
        
            
    def neuron_activation(self, X):
        # Return value of neuron activation function to X on every neuron
        activation_vals = np.zeros((X.shape[0], self.lattice_.neurons_nb_))

        for i, neuron in enumerate(self.lattice_.neurons_.values()):
            if not np.any(np.isnan(neuron.mean_)):
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

        # Corner case
        max_vals = np.max(responsibilities, axis=1)
        mask = max_vals < -nu
        responsibilities[mask] -= max_vals[mask, np.newaxis]
        
        responsibilities = np.exp(responsibilities)

        if self.use_weights_:
            for k in range(neurons_nb):
                responsibilities[:, k] *= neurons[neurons_indexes[k]].weight_

        return responsibilities / np.sum(responsibilities, axis=1, keepdims=True), log_neuron_activation


    def predict(self, X):
        neurons = self.lattice_.neurons_
        neurons_indexes = list(neurons.keys())

        y_pred = np.zeros(X.shape[0], dtype='int32')

        # E step
        responsibilities, partial_ll = self.find_responsibilities(X)

        # C step
        clusters = np.argmax(responsibilities, axis=1)

        for i, k in enumerate(clusters):
            y_pred[i] = neurons_indexes[k]
    
        # Save log_likelihood
        self.log_likelihood_ = np.sum(partial_ll[np.arange(X.shape[0]), clusters])
        
        return y_pred

    def update_nodes(self, X, clusters):
        # M-step
        # Updating all neuron based on batch mode

        # For numerical issues
        reg_covar = self.reg_covar_ * np.eye(self.n_features_in_)

        cluster_sums = np.zeros((self.lattice_.neurons_nb_, self.n_features_in_))
        cluster_counts = np.zeros(self.lattice_.neurons_nb_)
        
        neurons_keys = list(self.lattice_.neurons_.keys())
        to_idx_map = {key: index for index, key in enumerate(neurons_keys)}

        for i in range(X.shape[0]):
            idx = to_idx_map[clusters[i]]
            cluster_sums[idx] += X[i]
            cluster_counts[idx] += 1

        denominator = (self.H_ @ cluster_counts)
        mean = (self.H_ @ cluster_sums) / denominator[:, np.newaxis]

        if self.cov_type_ == 'full':
            for l, neuron_l in enumerate(self.lattice_.neurons_):
                self.lattice_.neurons_[neuron_l].mean_ = mean[l]
                cov = np.zeros((self.n_features_in_, self.n_features_in_))

                for k, neuron_k in enumerate(self.lattice_.neurons_):
                    diff = (X[clusters == neuron_k] - mean[l]).T
                    cov += self.H_[k, l] * np.dot(diff, diff.T)

                cov /= denominator[l]
                
                self.lattice_.neurons_[neuron_l].cov_ = cov + reg_covar
                self.lattice_.neurons_[neuron_l].weight_ = cluster_counts[l] / len(X)
        
        elif self.cov_type_ == 'diag':
            for l, neuron_l in enumerate(self.lattice_.neurons_):
                self.lattice_.neurons_[neuron_l].mean_ = mean[l]
                cov = np.zeros((self.n_features_in_, self.n_features_in_))

                for k, neuron_k in enumerate(self.lattice_.neurons_):
                    diff_squared = np.sum((X[clusters == neuron_k] - mean[l])**2, axis=0)
                    cov += np.diag(self.H_[k, l] * diff_squared)

                cov /= denominator[l]
                
                self.lattice_.neurons_[neuron_l].cov_ = cov + reg_covar * np.eye(self.n_features_in_)
                self.lattice_.neurons_[neuron_l].weight_ = cluster_counts[l] / len(X)
        
        elif self.cov_type_ == 'spherical':
            for l, neuron_l in enumerate(self.lattice_.neurons_):
                self.lattice_.neurons_[neuron_l].mean_ = mean[l]
                cov = np.zeros((self.n_features_in_, self.n_features_in_))

                for k, neuron_k in enumerate(self.lattice_.neurons_):
                    diff_squared = np.sum((X[clusters == neuron_k] - mean[l])**2, axis=0)
                    cov += self.H_[k, l] * diff_squared

                cov = cov.mean(1) / denominator[l]
                
                self.lattice_.neurons_[neuron_l].cov_ = (cov + reg_covar) * np.eye(self.n_features_in_)
                self.lattice_.neurons_[neuron_l].weight_ = cluster_counts[l] / len(X)


    def delete_empty_clusters(self):
        vertcies_for_deletion = []
        
        for vertex in self.lattice_.graph_.adj_list_:
            weight = self.lattice_.neurons_[vertex].weight_
            if weight == 0:
                vertcies_for_deletion.append(vertex)
        
        if vertcies_for_deletion:
            print("Empty clusters detected")

        for vertex in vertcies_for_deletion:
            self.lattice_.delete_vertex(vertex)
        
        self.H_ = np.eye(self.lattice_.neurons_nb_)

    
    #############################################################
    #############################################################
    ###################### BHATTACHARYYA ############################

    def delete_vertex_bhattacharyya(self):
        # Delete vertex by merging it with one of its 
        # neighbours based on min Bhattacharyya distance

        min_distance = np.inf
        best_pair = (None, None)

        vertecies = self.lattice_.graph_.adj_list_
        
        for vertex1 in vertecies:
            for vertex2 in vertecies[vertex1]:
                neuron1 = self.lattice_.neurons_[vertex1]
                neuron2 = self.lattice_.neurons_[vertex2]

                distance = self._bhattacharyya_distance(neuron1, neuron2)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (vertex1, vertex2)
        
        print("Best pair ->", best_pair)

        if np.exp(-min_distance) >= self.betta_:
            self.lattice_.collapse_edge(best_pair[0], best_pair[1])
            print("Merged -> {}, {}".format(best_pair[0], best_pair[1]))
            self.H_ = np.eye(self.lattice_.neurons_nb_)

            return True
        
        print("Not merged!")
        return False


    @staticmethod
    def _bhattacharyya_distance(neuron1:Neuron, neuron2:Neuron):
        mean_cov = 0.5 * (neuron1.cov_ + neuron2.cov_)

        return (neuron1.mean_ - neuron2.mean_).T @ np.linalg.inv(mean_cov) @ (neuron1.mean_ - neuron2.mean_) / 8 + \
        0.5 * np.log(np.linalg.det(mean_cov) / np.sqrt(np.linalg.det(neuron1.cov_) * np.linalg.det(neuron2.cov_)))
