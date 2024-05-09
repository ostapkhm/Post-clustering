import numpy as np
from copy import deepcopy
from scipy.stats import multivariate_normal

from .SOCEM import SOCEM
from .Utils import Neuron


class PbSOMClustering:
    def __init__(self, pbsom_model:SOCEM):
        self.model_ = pbsom_model
        # For EM
        self.neurons_ = deepcopy(pbsom_model.lattice_.neurons_)
        self.neurons_nb_ = pbsom_model.lattice_.neurons_nb_
        self.tol_ = pbsom_model.tol_
        self.reg_covar_ = pbsom_model.reg_covar_
        self.cov_type_ = pbsom_model.cov_type_
        self.n_features_in_ = None
    

    def fit(self, X, monitor=None):
        self.n_features_in_ = X.shape[1]

        ### First pass:
        # EM reduction
        self.em_reduction(X, monitor)

    

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


    def get_params(self):
        weights = np.empty(self.neurons_nb_)
        means = np.empty((self.neurons_nb_, self.n_features_in_))
        covs = np.empty((self.neurons_nb_, self.n_features_in_, self.n_features_in_))

        for i, (weight, mean, cov) in enumerate(self.neurons_.values()):
            weights[i] = weight
            means[i] = mean
            covs[i] = cov
        
        return weights, means, covs
    
    
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
        if monitor is not None:
            monitor.idx_ = best_monitor_idx_
