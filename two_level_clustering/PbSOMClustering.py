import numpy as np
from copy import deepcopy
from scipy.stats import multivariate_normal

from .SOCEM import SOCEM
from .Utils import Neuron, pcws_reg, multi_root, reestimate_params



class PbSOMClustering:
    def __init__(self, pbsom_model:SOCEM, merge_method, merge_threshold=None):
        self.model_ = pbsom_model
        # For EM
        self.neurons_ = pbsom_model.lattice_.neurons_
        self.neurons_nb_ = pbsom_model.lattice_.neurons_nb_
        self.tol_ = pbsom_model.tol_
        self.reg_covar_ = pbsom_model.reg_covar_
        self.cov_type_ = pbsom_model.cov_type_
        self.n_features_in_ = None
        # For merging
        self.merge_method_ = merge_method
        self.beta_ = merge_threshold
        self.bic_ = None

        self.labels_ = None
    

    def fit(self, X, monitor=None):
        self.n_features_in_ = X.shape[1]

        print("EM reduction started:")

        ### Zero pass:
        # monitors = self.model_.fit(X)
        
        ### First pass:
        # EM reduction
        self.em_reduction(X, monitor)

        ### Second pass:
        # Ridgeline / Entropy based
        if self.merge_method_ == 'ridgeline':
            self.labels_ = self.combine_gmm_ridgeline(X)
        elif self.merge_method_ == 'entropy':
            self.labels_ = self.combine_gmm_entropy(X, True)
    

    def cluster_activations(self, X):
        # Return log probability assignments for all points to clusters 
        activation_vals = np.zeros((X.shape[0], self.neurons_nb_))

        for i, neuron in enumerate(self.neurons_.values()):
            activation_vals[:, i] = multivariate_normal.pdf(X, neuron.mean_, neuron.cov_)

        # delta_min for numerical issues
        delta_min = 2.225e-308

        activation_vals[activation_vals < delta_min] = delta_min
        return activation_vals
    

    def find_responsibilities(self, X):
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


    def calculate_ll(self, X):
        # Calculate log likelihood
        activation_vals = self.cluster_activations(X)
        weights = np.array([neuron.weight_ for neuron in self.neurons_.values()])
        return np.sum(np.log(weights @ activation_vals.T))

    
    ####################################################
    ################# EM-reduction #####################
    ####################################################

    def em_reduction(self, X, monitor=None):
        self.n_features_in_ = X.shape[1]
        N = self.n_features_in_ * (self.n_features_in_ + 3) / 2
        neuron_indexes = list(self.neurons_.keys())
        
        weights = []
        for neuron in self.neurons_.values():
            weights.append(neuron.weight_)
        weights = np.array(weights)

        k = len(self.neurons_)
        k_min = 2
        best_model = None
        best_monitor_idx_ = 0
        times_saved = 0
        
        # The result of E-step
        current_llikelihood_ = self.calculate_ll(X)
        mindl = -current_llikelihood_ + 0.5 * N * np.sum(np.log(weights)) + 0.5 * (N + 1) * k * np.log(X.shape[0])
        prev_llikelihood_ = np.inf
        
        k_min_reached = False
        if monitor is not None:
            monitor.save(self.predict(X), False)
            times_saved += 1

        while not k_min_reached:            
            em_converged = False

            while not em_converged:
                # Inner loop
                current_comp = 0
                while current_comp < k:
                    # M-step with cluster deletion

                    responsibilities = self.find_responsibilities(X)
                    # Update parameters of neurons
                    neuron_idx = neuron_indexes[current_comp]
                    denominator = np.sum(responsibilities[:, current_comp]) 
                    self.neurons_[neuron_idx].mean_ = responsibilities[:, current_comp] @ X / denominator
                    diff = X - self.neurons_[neuron_idx].mean_
                    self.neurons_[neuron_idx].cov_ = np.dot((responsibilities[:, current_comp][:, np.newaxis] * diff).T, diff) / denominator + \
                                                     self.reg_covar_ * np.eye(self.n_features_in_)
                    
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
                        weights = np.delete(weights, current_comp)
                        self.neurons_nb_ -= 1
                        k -= 1

                        if monitor is not None:
                            monitor.save(self.predict(X), True)
                            times_saved+= 1

                    if not was_killed:
                        current_comp += 1
                
                prev_llikelihood_ = current_llikelihood_
                current_llikelihood_ = self.calculate_ll(X)
                description_length = -current_llikelihood_ + 0.5 * N * np.sum(np.log(weights)) + 0.5 * (N + 1) * k * np.log(X.shape[0])

                delta_llikelihood_ = current_llikelihood_ - prev_llikelihood_
                if np.abs(delta_llikelihood_/prev_llikelihood_) < self.tol_:
                    em_converged = True
                    if monitor is not None:
                        monitor.save(self.predict(X), False)
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

                if monitor is not None:
                    monitor.save(self.predict(X), True)
                    times_saved += 1
            else:
                k_min_reached = True
        
        self.neurons_ = best_model
        self.neurons_nb_ = len(best_model)
        monitor.idx_ = best_monitor_idx_


    ###########################################
    ########## Entropy based merging ##########
    ###########################################

    def combine_gmm_entropy(self, X, verbose=True):
        responsibilities = self.find_responsibilities(X)
        clusters_nb = self.neurons_nb_
        y_pred_history = np.zeros((clusters_nb, X.shape[0]))
        entropies = np.zeros(clusters_nb)
        merged_numbers = np.zeros(clusters_nb)

        neurons_indexes = list(self.neurons_.keys())
        y_pred = np.zeros(X.shape[0], dtype='int32')
        clusters = np.argmax(responsibilities, axis=1)

        for i, k in enumerate(clusters):
            y_pred[i] = neurons_indexes[k]
        
        entropies[0] = -np.sum(responsibilities * np.log(responsibilities))
        y_pred_history[0] = y_pred

        idx = 1
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
            y_pred_history[idx] = deepcopy(y_pred)
            merged_numbers[idx] = np.sum(y_pred == neurons_indexes[k])

            del neurons_indexes[k_p]
            entropies[idx] = -np.sum(responsibilities * np.log(responsibilities))
            
            idx += 1
            clusters_nb -= 1

        clusters_removed = pcws_reg(np.cumsum(merged_numbers), entropies, verbose)
        print("Clusters removed number:{}".format(clusters_removed))

        return y_pred_history[clusters_removed]
    

    ###########################################
    ######### Ridgeline based merging #########
    ###########################################

    def combine_gmm_ridgeline(self, X):
        y_pred = self.predict(X)
        ratios = np.zeros((self.neurons_nb_, self.neurons_nb_))
        key_to_idx = {key:index for index, key in enumerate(self.neurons_.keys())}

        # Precomputing
        for key1, neuron1 in self.neurons_.items():
            for key2, neuron2 in self.neurons_.items():
                idx1, idx2 = key_to_idx[key1], key_to_idx[key2]
                if key1 < key2 and not ratios[idx1, idx2]:
                    ratio = self.estimated_ratio(neuron1, neuron2)
                    ratios[idx1, idx2] = ratio
                    ratios[idx2, idx1] = ratio
        
        print("Precomputed!")

        while self.neurons_nb_ >= 2:
            max_ratio = -np.inf

            # Finding best pair
            for key1 in self.neurons_:
                for key2 in self.neurons_:
                    idx1, idx2 = key_to_idx[key1], key_to_idx[key2]

                    if ratios[idx1, idx2] > max_ratio:
                        max_ratio = ratios[idx1, idx2]
                        best_pair = (key1, key2)

            print("Best pair ->", best_pair)
            print("Max ratio ->", max_ratio)

            if max_ratio >= self.beta_:
                # Reestimate parameters of merged pair and save to smallest neuron key
                reestimate_params(self.neurons_[best_pair[0]], 
                                  self.neurons_[best_pair[1]])

                ratios[key_to_idx[best_pair[1]], :] = -np.inf
                ratios[:, key_to_idx[best_pair[1]]] = -np.inf

                # Remove the largest neuron key
                del self.neurons_[best_pair[1]]
                del key_to_idx[best_pair[1]]
                self.neurons_nb_ -= 1
    
                print("Merged -> {}, {}".format(best_pair[0], best_pair[1]))

                y_pred[y_pred == best_pair[1]] = best_pair[0]
                self.H_ = np.eye(self.neurons_nb_)

                # Update ratios for merged neuron and other neurons
                merged_neuron = self.neurons_[best_pair[0]]
                for key, neuron in self.neurons_.items():
                    if key != best_pair[0]:
                        ratio = self.estimated_ratio(merged_neuron, neuron)
                        idx1, idx2 = key_to_idx[best_pair[0]], key_to_idx[key]
                        ratios[idx1, idx2] = ratio
                        ratios[idx2, idx1] = ratio
                
            else:
                break
        
        return y_pred

    def estimated_ratio(self, model1:Neuron, model2:Neuron):
        # Check whether to merge two cluster based on r_val 
        alpha = model1.weight_ / (model1.weight_ + model2.weight_)
        dfunc = lambda x, model1, model2: self.piridge(x, model1, model2) - alpha

        roots = multi_root(dfunc, [0, 1], args=(model1, model2))

        if len(roots) == 1:
            return 1
        
        values = np.sort(self.f(roots, model1, model2))
        global_min, second_max = values[0], values[-2]

        return global_min / second_max


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


    def f(self, alpha:np.ndarray, model1:Neuron, model2:Neuron):
        X = self.to_ralpha(alpha, model1, model2)
        
        sum_prob = model1.weight_ + model2.weight_
        
        return model1.weight_ / sum_prob * multivariate_normal.pdf(X, mean=model1.mean_, cov=model1.cov_) + \
                model2.weight_ / sum_prob  * multivariate_normal.pdf(X, mean=model2.mean_, cov=model2.cov_)


    @staticmethod
    def to_ralpha(alpha:np.ndarray, model1:Neuron, model2:Neuron):
        X = np.empty((alpha.shape[0], model1.mean_.shape[0]))
        
        inv_cov_1 = np.linalg.inv(model1.cov_)
        inv_cov_2 = np.linalg.inv(model2.cov_)

        for i, a in enumerate(alpha):
            X[i] = np.linalg.inv((1 - a) * inv_cov_1 + a * inv_cov_2) @ \
                        ((1 - a) * inv_cov_1 @ model1.mean_ + a * inv_cov_2 @ model2.mean_)

        return X
