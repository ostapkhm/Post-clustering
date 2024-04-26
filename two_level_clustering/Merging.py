import numpy as np
from copy import deepcopy
from scipy.stats import multivariate_normal
from collections import OrderedDict

from .Utils import Neuron, pcws_reg, multi_root, reestimate_params


class Merging:
    def __init__(self, weights, means, covs, merge_method, merge_threshold=None):
        self.initialize_clusters(weights, means, covs)
        self.merge_method_ = merge_method
        self.beta_ = merge_threshold
        self.labels_ = None


    def initialize_clusters(self, weights, means, covs):
        self.clusters_ = OrderedDict()
        for i, (weight, mean, cov) in enumerate(zip(weights, means, covs)):
            self.clusters_[i] = Neuron(weight, mean, cov)
        
        self.clusters_nb_ = len(weights)

    def fit(self, X):
        self.n_features_in_ = X.shape[1]

        # Ridgeline / Entropy based merging
        if self.merge_method_ == 'ridgeline':
            self.labels_ = self.combine_gmm_ridgeline(X)
        elif self.merge_method_ == 'entropy':
            self.labels_ = self.combine_gmm_entropy(X, True)

    def cluster_activations(self, X):
        # Return log probability assignments for all points to clusters 
        activation_vals = np.zeros((X.shape[0], self.clusters_nb_))

        for i, neuron in enumerate(self.clusters_.values()):
            activation_vals[:, i] = multivariate_normal.pdf(X, neuron.mean_, neuron.cov_)

        # delta_min for numerical issues
        delta_min = 2.225e-308

        activation_vals[activation_vals < delta_min] = delta_min
        return activation_vals
    

    def find_responsibilities(self, X):
        # Unnormilized responsibilities

        # For numerical issues
        nu = 744

        neurons_indexes = list(self.clusters_.keys())
        responsibilities = np.log(self.cluster_activations(X))

        # Corner case
        max_vals = np.max(responsibilities, axis=1)
        mask = max_vals < -nu
        responsibilities[mask] -= max_vals[mask, np.newaxis]
        responsibilities = np.exp(responsibilities)

        for k in range(self.clusters_nb_):
            responsibilities[:, k] *= self.clusters_[neurons_indexes[k]].weight_

        return responsibilities / np.sum(responsibilities, axis=1, keepdims=True)


    def combine_gmm_entropy(self, X, verbose=True):
        responsibilities = self.find_responsibilities(X)
        clusters_nb = self.clusters_nb_
        y_pred_history = np.zeros((clusters_nb, X.shape[0]))
        entropies = np.zeros(clusters_nb)
        merged_numbers = np.zeros(clusters_nb)

        neurons_indexes = list(self.clusters_.keys())
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


    def combine_gmm_ridgeline(self, X):
        responsibilities = self.find_responsibilities(X)
        y_pred = np.zeros(X.shape[0], dtype='int32')
        clusters = np.argmax(responsibilities, axis=1)
        neurons_indexes = list(self.clusters_.keys())

        for i, k in enumerate(clusters):
            y_pred[i] = neurons_indexes[k]

        ratios = np.zeros((self.clusters_nb_, self.clusters_nb_))
        key_to_idx = {key:index for index, key in enumerate(self.clusters_.keys())}

        # Precomputing
        for key1, neuron1 in self.clusters_.items():
            for key2, neuron2 in self.clusters_.items():
                idx1, idx2 = key_to_idx[key1], key_to_idx[key2]
                if key1 < key2 and not ratios[idx1, idx2]:
                    ratio = self.estimated_ratio(neuron1, neuron2)
                    ratios[idx1, idx2] = ratio
                    ratios[idx2, idx1] = ratio
        
        print("Precomputed!")

        while self.clusters_nb_ >= 2:
            max_ratio = -np.inf

            # Finding best pair
            for key1 in self.clusters_:
                for key2 in self.clusters_:
                    idx1, idx2 = key_to_idx[key1], key_to_idx[key2]

                    if ratios[idx1, idx2] > max_ratio:
                        max_ratio = ratios[idx1, idx2]
                        best_pair = (key1, key2)

            print("Best pair ->", best_pair)
            print("Max ratio ->", max_ratio)

            if max_ratio >= self.beta_:
                # Reestimate parameters of merged pair and save to smallest neuron key
                reestimate_params(self.clusters_[best_pair[0]], 
                                  self.clusters_[best_pair[1]])

                ratios[key_to_idx[best_pair[1]], :] = -np.inf
                ratios[:, key_to_idx[best_pair[1]]] = -np.inf

                # Remove the largest neuron key
                del self.clusters_[best_pair[1]]
                del key_to_idx[best_pair[1]]
                self.clusters_nb_ -= 1
    
                print("Merged -> {}, {}".format(best_pair[0], best_pair[1]))

                y_pred[y_pred == best_pair[1]] = best_pair[0]
                self.H_ = np.eye(self.clusters_nb_)

                # Update ratios for merged neuron and other neurons
                merged_neuron = self.clusters_[best_pair[0]]
                for key, neuron in self.clusters_.items():
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
