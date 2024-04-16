from SOM import SOM
import numpy as np
from copy import deepcopy

from Monitor import Monitor
from Utils import Neuron
from scipy.stats import multivariate_normal
import matplotlib.pylab as plt
from scipy.optimize import root_scalar
import warnings

from copy import deepcopy
from typing import Callable, Iterable

class PbSOMClustering:
    def __init__(self, pbsom_model:SOM, merge_method, merge_threshold=None):
        self.model_ = pbsom_model

        # For EM
        self.neurons_ = pbsom_model.lattice_.neurons_
        self.neurons_nb_ = pbsom_model.lattice_.neurons_nb_
        self.tol_ = pbsom_model.tol_
        self.reg_covar_ = pbsom_model.reg_covar_
        self.cov_type_ = pbsom_model.cov_type_
        self.n_features_in_ = None

        # For merging
        self.merge_method_ = None
        self.betta_ = None
        self.bic_ = None

        if merge_method == 'ridgeline':
            self.merge_method_ = self.delete_vertex_ridgeline
        elif merge_method == 'entropy':
            self.merge_method_ = self.delete_vertex_entropy
    

    def fit(self, X):
        self.n_features_in_ = X.shape[1]

        print("BIC reduction started:")

        monitors = []
        ### Zero pass:
        # monitors = self.model_.fit(X)
        

        ### First pass:
        # EM + BIC clustering reduction
        deleted = True

        while deleted:
            monitor = Monitor(self)

            # Estimate parameters without a particular neuron
            idx_deleted = self.delete_verex(X, monitor)

            if idx_deleted is not None:
                monitors.append(monitor)
                
                print("Deleted {} cluster".format(idx_deleted))
                print(self.calculate_bic(X))
            else:
                deleted = False

        ### Second pass:
        # Ridgeline / Entropy based / Hierarchical

        return monitors

    def estimate_params(self, X: np.ndarray, monitor: Monitor = None):
        # Use CEM algorithm to estimate parameters

        iter_nb = 0

        while iter_nb < self.max_iter_:
            ### E and C-step ###
            clusters = self.predict(X)

            ### M-step ###
            self.update_nodes(X, clusters)

            if monitor is not None:
                monitor.save(self.predict(X))

            iter_nb += 1
    

    def cluster_activations(self, X):
        # Return probability assignments for all points to clusters 

        activation_vals = np.zeros((X.shape[0], self.neurons_nb_))

        for i, neuron in enumerate(self.neurons_.values()):
            activation_vals[:, i] = multivariate_normal.pdf(X, neuron.mean_, neuron.cov_)

        # delta_min for numerical issues
        delta_min = 2.225e-308

        activation_vals[activation_vals < delta_min] = delta_min
        return activation_vals
    

    def find_semi_responsibilities(self, X):
        # Unnormilized responsibilities

        # For numerical issues
        nu = 744

        neurons_indexes = list(self.neurons_.keys())
        responsibilities = np.log(self.cluster_activations(X))

        # Corner case
        max_vals = np.max(responsibilities, axis=1)
        mask = max_vals < -nu
        responsibilities[mask] -= max_vals[mask, np.newaxis]
        responsibilities = np.exp(responsibilities)

        for k in range(self.neurons_nb_):
            responsibilities[:, k] *= self.neurons_[neurons_indexes[k]].weight_

        return responsibilities

    def find_responsibilities(self, X):
        responsibilities = self.find_semi_responsibilities(X)
        return responsibilities / np.sum(responsibilities, axis=1, keepdims=True)
    

    def predict(self, X):
        neurons_indexes = list(self.neurons_.keys())
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
        # Updating all clusters based on batch mode

        n_features_in = X.shape[1]
        # For numerical issues
        reg_covar = self.reg_covar_ * np.eye(n_features_in)

        cluster_sums = np.zeros((self.neurons_nb_, n_features_in))
        cluster_counts = np.zeros(self.neurons_nb_)
        
        neurons_keys = list(self.neurons_.keys())
        to_idx_map = {key: index for index, key in enumerate(neurons_keys)}

        for i in range(X.shape[0]):
            idx = to_idx_map[clusters[i]]
            cluster_sums[idx] += X[i]
            cluster_counts[idx] += 1

        denominator = cluster_counts
        mean = cluster_sums / denominator[:, np.newaxis]

        for l, neuron_l in enumerate(self.neurons_):
            self.neurons_[neuron_l].mean_ = mean[l]

            diff = (X[clusters == neuron_l] - mean[l]).T
            cov = np.dot(diff, diff.T) / denominator[l]
            
            self.neurons_[neuron_l].cov_ = cov + reg_covar
            self.neurons_[neuron_l].weight_ = cluster_counts[l] / len(X)


    def calculate_log_likelihood(self, X):
        activation_vals = self.cluster_activations(X)
        weights = np.array([neuron.weight_ for neuron in self.neurons_.values()])
        return np.sum(np.log(weights @ activation_vals.T))


    def calculate_bic(self, X):
        n = X.shape[0]
        d = X.shape[1]
        k = self.neurons_nb_

        df = k * d * (d + 3) / 2
        log_likelihood = self.calculate_log_likelihood(X)
        
        return -log_likelihood + 0.5 * df * np.log(n)
    

    ####################################################
    ################ BIC-reduction #####################
    ####################################################

    def delete_verex(self, X: np.ndarray, monitor: Monitor):
        # Delete vertex based on BIC
        # if deleted -> return True, False otherwise
        
        if self.neurons_nb_ == 1:
            return None

        current_neurons = deepcopy(self.neurons_)
    
        current_bic = self.calculate_bic(X)
        best_candidate = None

        for neuron_key in current_neurons:
            # Delete a particular neuron
            del self.neurons_[neuron_key]
            self.neurons_nb_ -=1

            self.estimate_params(X)
            new_bic = self.calculate_bic(X)
            if new_bic < current_bic:
                current_bic = new_bic
                best_candidate = neuron_key

            # Revert changes
            self.neurons_ = deepcopy(current_neurons)
            self.neurons_nb_ += 1

        if best_candidate is not None:
            del self.neurons_[best_candidate]
            self.neurons_nb_ -=1
            self.estimate_params(X, monitor)

        print("Best candidate-> ", best_candidate)
        print("Best BIC-> ", current_bic)

        return best_candidate

    ####################################################
    ################# EM-reduction #####################
    ####################################################

    def em_reduction(self, X, eps):
        self.n_features_in_ = X.shape[1]
        N = self.n_features_in_ * (self.n_features_in_ + 3) / 2
        neuron_indexes = list(self.neurons_.keys())
        
        weights = []
        for neuron in self.neurons_.values():
            weights.append(neuron.weight_)
        weights = np.array(weights)

        k = len(self.neurons_)
        k_min = 2
        mindl = np.inf
        best_model = None
        best_monitor_idx_ = 0
        times_saved = 0
        
        # The result of E-step
        semi_responsibilities = self.find_semi_responsibilities(X)
        current_llikelihood_ =  np.sum(np.log(np.sum(semi_responsibilities, axis=1)))
        prev_llikelihood_ = np.inf
        
        k_min_reached = False
        monitor = Monitor(self)
        monitor.save(self.predict(X), None, False)

        while not k_min_reached:            
            em_converged = False

            while not em_converged:
                # Inner loop
                current_comp = 0
                while current_comp < k:
                    # M-step with cluster deletion

                    semi_responsibilities = self.find_semi_responsibilities(X)
                    responsibilities = semi_responsibilities / np.sum(semi_responsibilities, axis=1, keepdims=True)

                    # Update parameters of neurons
                    neuron_idx = neuron_indexes[current_comp]
                    denominator = np.sum(responsibilities[:, current_comp]) 
                    self.neurons_[neuron_idx].mean_ = responsibilities[:, current_comp] @ X / denominator
                    diff = X - self.neurons_[neuron_idx].mean_
                    self.neurons_[neuron_idx].cov_ = np.dot((responsibilities[:, current_comp][:, np.newaxis] * diff).T, diff) / denominator + self.reg_covar_ * np.eye(self.n_features_in_)
                    
                    # A part that is able to kill components
                    weights[current_comp] = np.maximum(0, np.sum(responsibilities[:, current_comp]) - N / 2) / X.shape[0]
                    weights = weights / np.sum(weights)

                    for p, current_neuron in enumerate(self.neurons_.values()):
                        current_neuron.weight_ = weights[p]

                    was_killed = False

                    if not weights[current_comp]:
                        print("Neuron: {} killed".format(neuron_idx))
                        was_killed = True
                        del self.neurons_[neuron_idx]
                        del neuron_indexes[current_comp]
                        self.neurons_nb_ -= 1
                        weights = np.delete(weights, current_comp)
                        k -= 1

                        monitor.save(self.predict(X), None, True)
                        times_saved+= 1

                    if not was_killed:
                        current_comp += 1
                
                semi_responsibilities = self.find_semi_responsibilities(X)
                prev_llikelihood_ = current_llikelihood_
                current_llikelihood_ = np.sum(np.log(np.sum(semi_responsibilities, axis=1)))
                description_length = -current_llikelihood_ + 0.5 * N * np.sum(np.log(weights)) + 0.5*(N + 1)*k*np.log(X.shape[0])

                delta_llikelihood_ = current_llikelihood_ - prev_llikelihood_

                if np.abs(delta_llikelihood_/prev_llikelihood_) < eps:
                    em_converged = True
                    monitor.save(self.predict(X), None, False)
                    times_saved += 1
            

            if description_length < mindl:
                mindl = description_length
                best_model = deepcopy(self.neurons_)
                best_monitor_idx_ = times_saved

            if k > k_min:
                current_comp = np.argmin(weights)
                neuron_idx = neuron_indexes[current_comp]

                print("Neuron: {} killed".format(neuron_idx))
                del self.neurons_[neuron_idx]
                del neuron_indexes[current_comp]
                self.neurons_nb_ -= 1
                weights = np.delete(weights, current_comp)
                weights /= np.sum(weights)
                k -= 1 

                monitor.save(self.predict(X), None, True)
                times_saved += 1
            else:
                k_min_reached = True
        
        self.neurons_ = best_model
        self.neurons_nb_ = len(best_model)
        monitor.idx_ = best_monitor_idx_ + 1

        return monitor


    ###########################################
    ########## Entropy based merging ##########
    ###########################################

    def combine_gmm_entropy(self, X):
        responsibilities = self.find_responsibilities(X)
        history = []

        entropies = []
        clusters_nb = self.neurons_nb_
        neurons_indexes = list(self.neurons_.keys())
        y_pred = np.zeros(X.shape[0], dtype='int32')
        clusters = np.argmax(responsibilities, axis=1)

        history.append(deepcopy(y_pred))

        for i, k in enumerate(clusters):
            y_pred[i] = neurons_indexes[k]
        
        entropies.append(-np.sum(responsibilities * np.log(responsibilities)))

        while clusters_nb > 1:

            max_diff_entropy = -np.inf
            best_pair = (None, None)

            for p in range(clusters_nb):
                for q in range(p + 1, clusters_nb):
                    combined_resp = responsibilities[:, p] + responsibilities[:, q]
                    
                    diff_entropy = -np.sum(responsibilities[:, p] * np.log(responsibilities[:, p]) + 
                                            responsibilities[:, q] * np.log(responsibilities[:, q])) + \
                                    + np.sum(combined_resp * np.log(combined_resp))

                    if diff_entropy > max_diff_entropy:
                        max_diff_entropy = diff_entropy
                        best_pair = (p, q)

            k, k_p = best_pair
            print("Best pair:{}, {} to merge".format(neurons_indexes[k], neurons_indexes[k_p]))

            combined_resp = responsibilities[:, k] + responsibilities[:, k_p]
            # Remain the smallest label and update assignmemts
            
            responsibilities = np.delete(responsibilities, k_p, axis=1)
            responsibilities[:, k] = combined_resp

            y_pred[y_pred == neurons_indexes[k_p]] = neurons_indexes[k]
            history.append(deepcopy(y_pred))
            number_of_merged_points = np.sum(y_pred == neurons_indexes[k])

            del neurons_indexes[k_p]
            entropies.append(-np.sum(responsibilities * np.log(responsibilities)))
            
            clusters_nb -= 1
        

        clusters_removed = self.pcws_reg(np.arange(len(entropies)), entropies, True)
        print("Clusters removed number:{}".format(clusters_removed))

        return history[clusters_removed]
    

    def pcws_reg(self, x, y, verbose=False):
        a1 = np.full((x.size - 2,), np.inf)
        a2 = np.full((x.size - 2,), np.inf)
        b1 = np.full((x.size - 2,), np.inf)
        b2 = np.full((x.size - 2,), np.inf)
        ss = np.full((x.size - 2,), np.inf)

        for c in range(1, x.size - 1):
            x1 = x[:c+1]
            y1 = y[:c+1]
            x2 = x[c:]
            y2 = y[c:]


            a1[c - 1] = (np.sum(x1 * y1) - np.sum(x1) * np.mean(y1)) / (np.sum(x1 ** 2) - np.sum(x1) ** 2 / x1.size)
            b1[c - 1] = -a1[c - 1] * np.mean(x1) + np.mean(y1)

            a2[c - 1] = (np.sum(x2 * y2) - np.mean(x2) * np.sum(y2)) / (np.sum(x2 ** 2) - np.sum(x2) ** 2 / x2.size)

            b2[c - 1] = -a2[c - 1] * np.mean(x2) + np.mean(y2)

            ss[c - 1] = np.sum((a1[c - 1] * x1 + b1[c - 1] - y1) ** 2) + np.sum((a2[c - 1] * x2 + b2[c - 1] - y2) ** 2)

        optimal_clust_nb = np.argmin(ss) + 1
        a1f = a1[optimal_clust_nb - 1]
        a2f = a2[optimal_clust_nb - 1]
        b1f = b1[optimal_clust_nb - 1]
        b2f = b2[optimal_clust_nb - 1]

        if verbose:
            plt.plot(x, y, 'ko')
            plt.plot(x[:optimal_clust_nb+1], a1f * x[:optimal_clust_nb+1] + b1f, 'r--')
            plt.plot(x[optimal_clust_nb:], a2f * x[optimal_clust_nb:] + b2f, 'r--')
            plt.show()

        return optimal_clust_nb

    ###########################################
    ######### Ridgeline based merging #########
    ###########################################

    # def delete_vertex_ridgeline(self):
    #     # Delete vertex by merging it with one of its 
    #     # neighbours based on max ridgeline

    #     max_ratio = -np.inf
    #     best_pair = (None, None)

    #     vertecies = self.lattice_.graph_.adj_list_

    #     # self.lattice_.graph_.show()
        
    #     # TODO Optimize(caching)!
    #     # TODO add ties

        
    #     for vertex1 in vertecies:
    #         for vertex2 in vertecies[vertex1]:
    #             neuron1 = self.lattice_.neurons_[vertex1]
    #             neuron2 = self.lattice_.neurons_[vertex2]

    #             if neuron1.weight_ != 0 and neuron2.weight_ != 0:
    #                 ratio = self.estimated_ratio(neuron1, neuron2)
    #                 if ratio > max_ratio:
    #                     max_ratio = ratio
    #                     best_pair = (vertex1, vertex2)
        
    #     print("Best pair ->", best_pair)

    #     if max_ratio >= self.betta_:
    #         self.lattice_.collapse_edge(best_pair[0], best_pair[1])
    #         print("Merged -> {}, {}".format(best_pair[0], best_pair[1]))

    #         self.H_ = np.eye(self.lattice_.neurons_nb_)
    #         return True

    #     return False



    # @staticmethod
    # def to_ralpha(alpha:np.ndarray, model1:Neuron, model2:Neuron):
    #     X = np.empty((alpha.shape[0], model1.mean_.shape[0]))
        
    #     inv_cov_1 = np.linalg.inv(model1.cov_)
    #     inv_cov_2 = np.linalg.inv(model2.cov_)

    #     for i, a in enumerate(alpha):
    #         X[i] = np.linalg.inv((1 - a) * inv_cov_1 + a * inv_cov_2) @ \
    #                     ((1 - a) * inv_cov_1 @ model1.mean_ + a * inv_cov_2 @ model2.mean_)

    #     return X


    # def piridge(self, alpha, model1, model2):
    #     if isinstance(alpha, float):
    #         alpha = np.array([alpha])

    #     X = self.to_ralpha(alpha, model1, model2)

    #     delta_min = 2.225e-300
    #     phi1 = multivariate_normal.pdf(X, mean=model1.mean_, cov=model1.cov_) + delta_min
    #     phi2 = multivariate_normal.pdf(X, mean=model2.mean_, cov=model2.cov_) + delta_min

    #     if not isinstance(phi1, np.ndarray):
    #         phi1 = np.array([phi1])
    #         phi2 = np.array([phi2])
        
    #     numerator = alpha * phi1
    #     denominator = (1 - alpha) * phi2
    #     mask = denominator > 0

    #     numerator = numerator[mask]
    #     denominator = denominator[mask]

    #     res = 1 / (1 + numerator / denominator)
        
    #     return np.concatenate((res, np.zeros(len(alpha) - np.sum(mask))))



    # @staticmethod
    # def multi_root(f: Callable, bracket: Iterable[float], args: Iterable = (), n: int = 500) -> np.ndarray:
    #     # Evaluate function in given bracket
    #     x = np.linspace(*bracket, n)
    #     y = f(x, *args)

    #     # Find where adjacent signs are not equal
    #     sign_changes = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]

    #     # Find roots around sign changes
    #     root_finders = (
    #         root_scalar(
    #             f=f,
    #             args=args,
    #             bracket=(x[s], x[s+1])
    #         )
    #         for s in sign_changes
    #     )

    #     roots = np.array([
    #         r.root if r.converged else np.nan
    #         for r in root_finders
    #     ])

    #     if np.any(np.isnan(roots)):
    #         warnings.warn("Not all root finders converged for estimated brackets! Maybe increase resolution `n`.")
    #         roots = roots[~np.isnan(roots)]

    #     roots_unique = np.unique(roots)
    #     if len(roots_unique) != len(roots):
    #         warnings.warn("One root was found multiple times. "
    #                     "Try to increase or decrease resolution `n` to see if this warning disappears.")

    #     return roots_unique


    # def f(self, alpha:np.ndarray, model1:Neuron, model2:Neuron):
    #     X = self.to_ralpha(alpha, model1, model2)
        
    #     sum_prob = model1.weight_ + model2.weight_
        
    #     return model1.weight_ / sum_prob * multivariate_normal.pdf(X, mean=model1.mean_, cov=model1.cov_) + \
    #             model2.weight_ / sum_prob  * multivariate_normal.pdf(X, mean=model2.mean_, cov=model2.cov_)


    # def estimated_ratio(self, model1:Neuron, model2:Neuron):
    #     # Check whether to merge two cluster based on r_val 
    #     alpha = model1.weight_ / (model1.weight_ + model2.weight_)
    #     dfunc = lambda x, model1, model2: self.piridge(x, model1, model2) - alpha

    #     roots = self.multi_root(dfunc, [0, 1], args=(model1, model2))

    #     if len(roots) == 1:
    #         return 1
        
    #     values = np.sort(self.f(roots, model1, model2))
    #     global_min, second_max = values[0], values[-2]

    #     return global_min / second_max