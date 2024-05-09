from graph.Utils import Lattice
from graph.Utils import Neuron
from graph.SOM import SOM

from graph.Monitor import Monitor
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.optimize import root_scalar
import warnings
import numpy as np

from copy import deepcopy

from typing import Callable, Iterable



class SOCEM(SOM):
    def __init__(self, lattice: Lattice, sigma_start, sigma_step, betta, cov_type, method, tol=1e-4, max_iter=100, smoothing_factor = 50, use_weights=False, random_state=None, reg_covar=1e-6):
        super().__init__(lattice, sigma_start, sigma_step, tol, max_iter, use_weights, random_state, reg_covar)
        self.betta_ = betta
        self.merge_method_ = None
        self.smoothing_factor_ = smoothing_factor
        self.cov_type_ = cov_type
        self.criteria_ = None

        if method == 'bhattacharyya':
            self.merge_method_ = self.delete_vertex_bhattacharyya
        elif method == 'ridgeline':
            self.merge_method_ = self.delete_vertex_ridgeline
        elif method == 'mdl':
            self.merge_method_ = self.delete_vertex_mdl
        elif method == 'entropy':
            self.merge_method_ = self.delete_vertex_entropy
        

    def fit(self, X: np.ndarray, return_monitors=False):
        map_changed = True
        monitors = []
        use_map = False

        self.sigma_ = self.sigma_start_

        while map_changed:
            print("Neurons->", self.lattice_.neurons_.keys())

            ### Estimate parameters of current map ###
            monitor = Monitor(self)
            self.estimate_params(X, use_map, monitor)
            y_pred = self.predict(X)
            monitor.save(y_pred)


            ### Delete empty clusters ###
            self.delete_empty_clusters()
            self.criteria_ = self.calculate_mdl(X)
            monitor.save(y_pred, True, True)
            
            ### Reconstruct map ###
            self.reconstruct_map()
            y_pred = self.predict(X)
            monitor.save(y_pred, True)

            ### Delete unnecessary vertex based on some criteria ###
            map_changed = self.merge_method_(X)
            y_pred = self.predict(X)
            monitor.save(y_pred, True)

            monitors.append(monitor)
            
            print("Map changed -> ", map_changed)
            
            print("------------------")
            self.lattice_.graph_.show()
            print("------------------")

            if not use_map:
                use_map = True
                self.use_weights_ = True
                self.cov_type_ = 'full'
            
            self.sigma_ = self.smoothing_factor_ * self.sigma_step_
        
        return monitors

    def estimate_params(self, X: np.ndarray, use_map, monitor=None):
        self.n_features_in_ = X.shape[1]
        neurons_nb = self.lattice_.neurons_nb_
        neurons = self.lattice_.neurons_

        if not use_map:
            ### Initial estimates using KMeans if map was not initialized ###
            k_means = KMeans(n_clusters=neurons_nb, random_state=self.random_state, n_init='auto')
            k_means.fit(X)

            # For numerical issues
            reg_covar = self.reg_covar_ * np.eye(self.n_features_in_)

            for i, idx in enumerate(neurons):
                idxs = (k_means.labels_ == i)

                neurons[idx].weight_ = np.sum(idxs) / X.shape[0]
                neurons[idx].mean_ = k_means.cluster_centers_[i]
                neurons[idx].cov_ = np.cov(X[idxs], rowvar=False) + reg_covar

        else:
            # No interaction with others = CEM algorithm
            self.lattice_.pairwise_distance_ = np.full((neurons_nb, neurons_nb), np.inf)
            np.fill_diagonal(self.lattice_.pairwise_distance_, 0)  
        
        self.H_ = np.exp(-0.5 * self.lattice_.pairwise_distance_ / self.sigma_**2)

        if monitor is not None:
            if not use_map:
                monitor.save(k_means.predict(X))
            else:
                monitor.save(self.predict(X))


        #############################################
        iter_nb = 0

        while np.exp(-0.5 / self.sigma_**2) > 0 and iter_nb < self.max_iter_:
            self.sigma_ -= self.sigma_step_
            self.H_ = np.exp(-0.5 * self.lattice_.pairwise_distance_ / self.sigma_**2)

            ### E and C-step ###
            clusters = self.predict(X)

            ### M-step ###
            self.update_nodes(X, clusters)

            if monitor is not None:
                monitor.save(self.predict(X))

            iter_nb += 1


    def neuron_activation(self, X):
        # Return value of neuron activation function to X on every neuron
        activation_vals = np.zeros((X.shape[0], self.lattice_.neurons_nb_))

        for i, neuron in enumerate(self.lattice_.neurons_.values()):
            if not np.all(np.isnan(neuron.mean_)):
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
        neurons_nb = self.lattice_.neurons_nb_
        avg_ll_matrix = None
        occurrences = None

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

        #print("res->", np.exp(-0.5 / self.sigma_**2))
        
        # print("H->", self.H_)
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



    
    #############################################################
    #############################################################
    ###################### BHATTACHARYYA ############################

    def delete_vertex_bhattacharyya(self, X=None):
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
            return True
        
        return False


    @staticmethod
    def _bhattacharyya_distance(neuron1:Neuron, neuron2:Neuron):
        mean_cov = 0.5 * (neuron1.cov_ + neuron2.cov_)

        return (neuron1.mean_ - neuron2.mean_).T @ np.linalg.inv(mean_cov) @ (neuron1.mean_ - neuron2.mean_) / 8 + \
        0.5 * np.log(np.linalg.det(mean_cov) / np.sqrt(np.linalg.det(neuron1.cov_) * np.linalg.det(neuron2.cov_)))


    #############################################################
    #############################################################
    ######################## RIDGELINE ##########################

    def delete_vertex_ridgeline(self, X=None):
        # Delete vertex by merging it with one of its 
        # neighbours based on max ridgeline

        max_ratio = -np.inf
        best_pair = (None, None)

        vertecies = self.lattice_.graph_.adj_list_

        # self.lattice_.graph_.show()
        
        # TODO Optimize(caching)!
        # TODO add ties

        
        for vertex1 in vertecies:
            for vertex2 in vertecies[vertex1]:
                neuron1 = self.lattice_.neurons_[vertex1]
                neuron2 = self.lattice_.neurons_[vertex2]

                if neuron1.weight_ != 0 and neuron2.weight_ != 0:
                    ratio = self.estimated_ratio(neuron1, neuron2)
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_pair = (vertex1, vertex2)
        
        print("Best pair ->", best_pair)

        if max_ratio >= self.betta_:
            self.lattice_.collapse_edge(best_pair[0], best_pair[1])
            print("Merged -> {}, {}".format(best_pair[0], best_pair[1]))

            self.H_ = np.exp(-0.5 * self.lattice_.pairwise_distance_ / self.sigma_**2)
            return True

        return False



    @staticmethod
    def to_ralpha(alpha:np.ndarray, model1:Neuron, model2:Neuron):
        X = np.empty((alpha.shape[0], model1.mean_.shape[0]))
        
        inv_cov_1 = np.linalg.inv(model1.cov_)
        inv_cov_2 = np.linalg.inv(model2.cov_)

        for i, a in enumerate(alpha):
            X[i] = np.linalg.inv((1 - a) * inv_cov_1 + a * inv_cov_2) @ \
                        ((1 - a) * inv_cov_1 @ model1.mean_ + a * inv_cov_2 @ model2.mean_)

        return X


    def piridge(self, alpha, model1, model2):
        if isinstance(alpha, float):
            alpha = np.array([alpha])

        X = self.to_ralpha(alpha, model1, model2)

        delta_min = 2.225e-300
        phi1 = multivariate_normal.pdf(X, mean=model1.mean_, cov=model1.cov_) + delta_min
        phi2 = multivariate_normal.pdf(X, mean=model2.mean_, cov=model2.cov_) + delta_min

        if not isinstance(phi1, np.ndarray):
            phi1 = np.array([phi1])
            phi2 = np.array([phi2])
        
        numerator = alpha * phi1
        denominator = (1 - alpha) * phi2
        mask = denominator > 0

        numerator = numerator[mask]
        denominator = denominator[mask]

        res = 1 / (1 + numerator / denominator)
        
        return np.concatenate((res, np.zeros(len(alpha) - np.sum(mask))))



    @staticmethod
    def multi_root(f: Callable, bracket: Iterable[float], args: Iterable = (), n: int = 500) -> np.ndarray:
        # Evaluate function in given bracket
        x = np.linspace(*bracket, n)
        y = f(x, *args)

        # Find where adjacent signs are not equal
        sign_changes = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]

        # Find roots around sign changes
        root_finders = (
            root_scalar(
                f=f,
                args=args,
                bracket=(x[s], x[s+1])
            )
            for s in sign_changes
        )

        roots = np.array([
            r.root if r.converged else np.nan
            for r in root_finders
        ])

        if np.any(np.isnan(roots)):
            warnings.warn("Not all root finders converged for estimated brackets! Maybe increase resolution `n`.")
            roots = roots[~np.isnan(roots)]

        roots_unique = np.unique(roots)
        if len(roots_unique) != len(roots):
            warnings.warn("One root was found multiple times. "
                        "Try to increase or decrease resolution `n` to see if this warning disappears.")

        return roots_unique


    def f(self, alpha:np.ndarray, model1:Neuron, model2:Neuron):
        X = self.to_ralpha(alpha, model1, model2)
        
        sum_prob = model1.weight_ + model2.weight_
        
        return model1.weight_ / sum_prob * multivariate_normal.pdf(X, mean=model1.mean_, cov=model1.cov_) + \
                model2.weight_ / sum_prob  * multivariate_normal.pdf(X, mean=model2.mean_, cov=model2.cov_)


    def estimated_ratio(self, model1:Neuron, model2:Neuron):
        # Check whether to merge two cluster based on r_val 
        alpha = model1.weight_ / (model1.weight_ + model2.weight_)
        dfunc = lambda x, model1, model2: self.piridge(x, model1, model2) - alpha

        roots = self.multi_root(dfunc, [0, 1], args=(model1, model2))

        if len(roots) == 1:
            return 1
        
        values = np.sort(self.f(roots, model1, model2))
        global_min, second_max = values[0], values[-2]

        return global_min / second_max

    
    #########################################################
    #################    Reconstruct map  ###################
    #########################################################

    def delete_empty_clusters(self):
        vertcies_for_deletion = []
        
        for vertex in self.lattice_.graph_.adj_list_:
            weight = self.lattice_.neurons_[vertex].weight_
            if weight == 0:
                vertcies_for_deletion.append(vertex)
        
        for vertex in vertcies_for_deletion:
            self.lattice_.delete_vertex(vertex)
        
        self.H_ = np.exp(-0.5 * self.lattice_.pairwise_distance_ / self.sigma_**2)


    def reconstruct_map(self):
        # Reconstruct map by rewiring to k nearest neighbours using js_divergence
        self.lattice_.reconstruct_map()



    # #############################################################
    # #############################################################
    # ######################    MDL    ############################


    def delete_vertex_mdl(self, X):
        # Brute force all vertices in graph to find the best vertex for deletion

        if self.lattice_.neurons_nb_ == 1:
            return False

        current_mdl = self.calculate_mdl(X)
        current_lattice = deepcopy(self.lattice_)

        best_mdl = current_mdl
        best_vertex = None

        for v in current_lattice.graph_.adj_list_:
            self.lattice_ = deepcopy(current_lattice)
            self.lattice_.delete_vertex(v)

            self.sigma_ = self.smoothing_factor_ * self.sigma_step_
            
            print("Estimate params without vertex", v)
            self.estimate_params(X, True, None)

            print("Neurons->", self.lattice_.neurons_.keys())

            new_mdl = self.calculate_mdl(X)
            print("New MDL:", new_mdl)

            if new_mdl < best_mdl:
                best_vertex = v
                best_mdl = new_mdl

        print("Best vertex->", best_vertex)
        print('Current MDL->', current_mdl)
        print('Best MDL->', best_mdl)

        self.lattice_ = deepcopy(current_lattice)
        if best_mdl < current_mdl:
            # Accept deletion
            self.lattice_.delete_vertex(best_vertex)

            print("Deleted -> {}".format(best_vertex))
            return True
        
        return False
    

    def calculate_mdl(self, X):
        n = X.shape[0]
        d = X.shape[1]

        k = self.lattice_.neurons_nb_
        log_likelihood = self.calc_log_likelihood(X)

        # return 0.5 * d * np.sum(np.log(n * weights / 12)) + 0.5 * k * (np.log(n / 12) + d + 1) - log_likelihood
    
        df = k * d * (d + 3) / 2

        return -log_likelihood + 0.5 * df * np.log(n)
    
    # #############################################################
    # #############################################################
    # ######################  Entropy based   #####################

    def delete_vertex_entropy(self, X=None):
        # Delete vertex by merging it with one of its 
        # neighbours based on min entropy

        max_dif= -np.inf
        best_pair = (None, None)
        best_pair_idx = (None, None)

        neurons = self.lattice_.neurons_
        neurons_indexes = list(neurons.keys())

        if len(neurons) == 1:
            return False

        responsibilities, _ = self.find_responsibilities(X)
        
        for v1_idx in range(0, len(neurons)):
            for v2_idx in range(v1_idx + 1, len(neurons)):
                diff = self.estimated_entropy_diff(responsibilities, v1_idx, v2_idx)
                if diff > max_dif:
                    max_dif = diff
                    best_pair = (neurons_indexes[v1_idx], neurons_indexes[v2_idx])
                    best_pair_idx = (v1_idx, v2_idx)
        
        print("Best pair ->", best_pair)
        
        self.criteria_ = self.calculate_entropy(X, best_pair_idx)

        self.lattice_.collapse_edge(best_pair[0], best_pair[1])
        print("Merged -> {}, {}".format(best_pair[0], best_pair[1]))
        self.H_ = np.exp(-0.5 * self.lattice_.pairwise_distance_ / self.sigma_**2)

        return True



    def estimated_entropy_diff(self, resp, neuron1_idx, neuron2_idx):
        return -np.sum(resp[:, neuron1_idx] * np.log(resp[:, neuron1_idx]) + resp[:, neuron2_idx] * np.log(resp[:, neuron2_idx])) + \
            np.sum((resp[:, neuron1_idx] + resp[:, neuron2_idx]) * np.log(resp[:, neuron1_idx] + resp[:, neuron2_idx]))

    
    def calculate_entropy(self, X, pair=None):
        self.use_weights_ = True

        responsibilities, _ = self.find_responsibilities(X)

        if pair is None:
            return -np.sum(responsibilities * np.log(responsibilities))
        else:
            return self.calculate_entropy_pair(responsibilities, pair[0], pair[1])
    

    def calculate_entropy_pair(self, resp, neuron1_idx, neuron2_idx):
        pair_entropy = (resp[:, neuron1_idx] + resp[:, neuron2_idx]) * np.log(resp[:, neuron1_idx] + resp[:, neuron2_idx])

        return -np.sum(np.sum(resp * np.log(resp), axis=1) - resp[:, neuron1_idx] * np.log(resp[:, neuron1_idx])
                  - resp[:, neuron2_idx] * np.log(resp[:, neuron2_idx]) + pair_entropy)


    
    def calc_log_likelihood(self, X):
        activation_vals = self.neuron_activation(X)
        weights = np.array([neuron.weight_ for neuron in self.lattice_.neurons_.values()])
        return np.sum(np.log(weights @ activation_vals.T))


    # def calculate_entropy_ratio(self, vertex, X):
    #     d = X.shape[1]
    #     mean = self.lattice_.neurons_[vertex].mean_
    #     cov = self.lattice_.neurons_[vertex].cov_

    #     max_entropy = -0.5 * (d * (1 + np.log(2*np.pi)) + np.log(np.linalg.det(cov)))

    #     y_pred = self.predict(X)
    #     mask = y_pred == vertex
    #     clusters_points = X[mask]

    #     entropy = -np.mean(multivariate_normal.logpdf(clusters_points, mean, cov))

    #     return entropy / max_entropy
    

    # def edge_cutting(self, X):
    #     neurons_nb = self.lattice_.neurons_nb_
    #     neurons_keys = list(self.lattice_.neurons_.keys())
    #     to_idx_map = {key: index for index, key in enumerate(neurons_keys)}

    #     print("Current_nb->", neurons_nb)
    #     print("Current_vertices->", neurons_keys)

    #     adj_list = self.lattice_.graph_.adj_list_

    #     _, avg_ll_matrix = self.predict(X, return_ll_matrix=True)
    #     h = np.max(-np.diag(avg_ll_matrix))

    #     divergence_matrix = np.zeros((neurons_nb, neurons_nb))
    #     for k in range(neurons_nb):
    #         divergence_matrix[k] = avg_ll_matrix[k, k] - avg_ll_matrix[k]

    #     for e in adj_list:
    #         for v in adj_list[e]:
    #             idx_e = to_idx_map[e]
    #             idx_v = to_idx_map[v]

    #             if 0.5 * (divergence_matrix[idx_e, idx_v] + divergence_matrix[idx_v, idx_e]) > self.betta_ * h:
    #                 print("Edge deleted:{}->{}".format(e, v))
    #                 self.lattice_.delete_edge(e, v)
        
    #     self.lattice_.update_distances()
        

    # def vertex_deleting(self, X):
    #     redundant_vertex = None
    #     min_mdl = self.mdl_
        
    #     for neuron_nb in self.lattice_.neurons_:
    #         # Estimate parameters of M - {m} clusters using CEM

    #         mdl_candidate = self.CEM_without_vertex(X, neuron_nb)
    #         if mdl_candidate < min_mdl:
    #             redundant_vertex = neuron_nb
    #             min_mdl = mdl_candidate
        
    #     if redundant_vertex is not None:
    #         print("Min MDL->", min_mdl)
    #         print("Deleted vertex:{}".format(redundant_vertex))
    #         self.lattice_.delete_vertex(redundant_vertex)
        
    #     return redundant_vertex is not None
    