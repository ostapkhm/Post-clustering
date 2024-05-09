import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

from .Utils import Neuron, Lattice


class SOCEM:
    def __init__(self, lattice: Lattice, sigma_start, sigma_step, betta, cov_type, tol=1e-4, max_iter=100, use_weights=False, random_state=None, reg_covar=1e-6, verbose=False):
        # For SOCEM
        self.lattice_ = lattice
        self.sigma_start_ = sigma_start
        self.sigma_step_ = sigma_step
        self.H_ = None 
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.use_weights_ = use_weights
        self.cov_type_ = cov_type
        self.reg_covar_ = reg_covar
        # For bhattacharyya merging
        self.betta_ = betta
        self.merge_method_ = None
        
        self.n_features_in_ = None
        self.random_state = random_state
        self.verbose_ = verbose
        

    def fit(self, X: np.ndarray, monitor=None, verbose=False):
        self.estimate_params(X, monitor)
        y_pred = self.predict(X)
        
        if monitor is not None:
            monitor.save(y_pred)

        ### Delete empty clusters ###
        self.delete_empty_clusters()
        if monitor is not None:
            monitor.save(y_pred, True)

        map_changed = True
        while map_changed:
            if self.verbose_:
                print("Neurons->", self.lattice_.neurons_.keys())
            ### Merge vertices based on bhattacharyya distance ###
            map_changed = self.delete_vertex_bhattacharyya()
            self.delete_empty_clusters()
            y_pred = self.predict(X)

            if monitor is not None:
                monitor.save(y_pred, True)
                


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

        if monitor is not None:
            monitor.save(k_means.predict(X))
        #############################################

        prev_ll = -np.inf
        sigma = self.sigma_start_
        while sigma > 0:
            self.H_ = np.exp(-0.5 * self.lattice_.pairwise_distance_ / sigma**2)

            current_ll = self.caluclate_ll(X)
            iter_nb = 0
            while iter_nb < self.max_iter_ and current_ll - prev_ll > self.tol_:
                # E step
                responsibilities = self.find_responsibilities(X)
                # C step
                clusters = np.argmax(responsibilities, axis=1)
                ### M-step ###
                self.update_nodes(X, clusters)

                prev_ll = current_ll
                current_ll = self.caluclate_ll(X)
                iter_nb += 1

            if monitor is not None:
                monitor.save(self.predict(X))
            
            sigma -= self.sigma_step_
        
            
    def neuron_log_activation(self, X):
        # Return value of neuron activation function to X on every neuron
        activation_vals = np.zeros((X.shape[0], self.lattice_.neurons_nb_))

        for i, neuron in enumerate(self.lattice_.neurons_.values()):
            if not np.any(np.isnan(neuron.mean_)):
                activation_vals[:, i] = multivariate_normal.logpdf(X, neuron.mean_, neuron.cov_)

        return activation_vals


    def find_responsibilities(self, X):
        # E-step

        # For numerical issues
        nu = 744

        neurons = self.lattice_.neurons_
        neurons_nb = self.lattice_.neurons_nb_
        neurons_indexes = list(neurons.keys())

        responsibilities = self.neuron_log_activation(X) @ self.H_

        # Corner case
        max_vals = np.max(responsibilities, axis=1)
        mask = max_vals < -nu
        responsibilities[mask] -= max_vals[mask, np.newaxis]
        
        responsibilities = np.exp(responsibilities)

        if self.use_weights_:
            for k in range(neurons_nb):
                responsibilities[:, k] *= neurons[neurons_indexes[k]].weight_

        return responsibilities / np.sum(responsibilities, axis=1, keepdims=True)


    def predict(self, X):
        neurons = self.lattice_.neurons_
        neurons_indexes = list(neurons.keys())
        y_pred = np.zeros(X.shape[0], dtype='int32')

        # E step
        responsibilities = self.find_responsibilities(X)
        # C step
        clusters = np.argmax(responsibilities, axis=1)

        for i, k in enumerate(clusters):
            y_pred[i] = neurons_indexes[k]

        return y_pred


    def update_nodes(self, X, clusters):
        # M-step
        # Updating all neuron based on batch mode

        cluster_sums = np.zeros((self.lattice_.neurons_nb_, self.n_features_in_))
        cluster_counts = np.zeros(self.lattice_.neurons_nb_)

        np.add.at(cluster_counts, clusters, 1)
        for i in range(self.lattice_.neurons_nb_):
            cluster_sums[i] = np.sum(X[clusters == i], axis=0)

        denominator = (self.H_ @ cluster_counts)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = (self.H_ @ cluster_sums) / denominator[:, np.newaxis]

        for l, neuron_l in enumerate(self.lattice_.neurons_):
            cov = np.zeros((self.n_features_in_, self.n_features_in_))

            for k, neuron_k in enumerate(self.lattice_.neurons_):
                diff = X[clusters == neuron_k] - mean[l]

                if self.cov_type_ == 'full':
                    cov += self.H_[k, l] * np.dot(diff.T, diff)
                elif self.cov_type_ == 'diag':
                    diff_squared = np.sum(diff**2, axis=0)
                    cov += np.diag(self.H_[k, l] * diff_squared)
                elif self.cov_type_ == 'spherical':
                    diff_squared = np.sum(diff**2, axis=0)
                    cov += self.H_[k, l] * diff_squared

            if self.cov_type_ == 'spherical':
                cov = cov.mean(axis=1)

            cov /= denominator[l]

            if self.cov_type_ == 'spherical':
                cov = (cov + self.reg_covar_) * np.eye(self.n_features_in_)
            else:
                cov = cov + self.reg_covar_ * np.eye(self.n_features_in_)

            
            self.lattice_.neurons_[neuron_l].weight_ = cluster_counts[l] / len(X)
            self.lattice_.neurons_[neuron_l].mean_ = mean[l]
            self.lattice_.neurons_[neuron_l].cov_ = cov


    def delete_empty_clusters(self):
        vertcies_for_deletion = []
        
        for vertex in self.lattice_.graph_.adj_list_:
            weight = self.lattice_.neurons_[vertex].weight_
            if weight == 0:
                vertcies_for_deletion.append(vertex)
        
        if self.verbose_:
            if vertcies_for_deletion:
                print("Empty clusters detected")

        for vertex in vertcies_for_deletion:
            self.lattice_.delete_vertex(vertex)
        
        self.H_ = np.eye(self.lattice_.neurons_nb_)
    
    def caluclate_ll(self, X):
        # Calculate classification log likelihood
        semi_responsibilities = self.neuron_log_activation(X) @ self.H_
        return np.sum(np.max(semi_responsibilities, axis=1))
    
    #############################################################
    #############################################################
    ###################### BHATTACHARYYA ########################

    def delete_vertex_bhattacharyya(self):
        # Delete vertex by merging it with one of its 
        # neighbours based on minimum Bhattacharyya distance

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
        
        if self.verbose_:
            print("Best pair ->", best_pair)

        if np.exp(-min_distance) >= self.betta_:
            self.lattice_.collapse_edge(best_pair[0], best_pair[1])

            if self.verbose_:
                print("Merged -> {}, {}".format(best_pair[0], best_pair[1]))
            self.H_ = np.eye(self.lattice_.neurons_nb_)

            return True
    
        return False


    @staticmethod
    def _bhattacharyya_distance(neuron1:Neuron, neuron2:Neuron):
        mean_cov = 0.5 * (neuron1.cov_ + neuron2.cov_)

        return (neuron1.mean_ - neuron2.mean_).T @ np.linalg.inv(mean_cov) @ (neuron1.mean_ - neuron2.mean_) / 8 + \
        0.5 * np.log(np.linalg.det(mean_cov) / np.sqrt(np.linalg.det(neuron1.cov_) * np.linalg.det(neuron2.cov_)))
