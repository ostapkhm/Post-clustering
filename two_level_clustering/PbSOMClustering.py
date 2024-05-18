import numpy as np
from copy import deepcopy
from .Utils import compute_activations, compute_responsibilities

from .SOCEM import SOCEM


class PbSOMClustering:
    def __init__(self, pbsom_model: SOCEM):
        self.model_ = pbsom_model
        self.neurons_ = deepcopy(pbsom_model.lattice_.neurons_)
        self.neurons_nb_ = pbsom_model.lattice_.neurons_nb_
        self.tol_ = pbsom_model.tol_
        self.reg_covar_ = pbsom_model.reg_covar_
        self.cov_type_ = pbsom_model.cov_type_
        self.n_features_in_ = None

    def fit(self, X, monitor=None):
        self.n_features_in_ = X.shape[1]
        self.em_reduction(X, monitor)

    def neuron_activations(self, X):
        return compute_activations(X, self.neurons_, self.neurons_nb_)

    def find_responsibilities(self, X):
        return compute_responsibilities(X, self.neurons_, self.neurons_nb_)

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
        activation_vals = self.neuron_activations(X)
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

    def em_reduction(self, X, monitor=None):
        self.n_features_in_ = X.shape[1]
        N = self.n_features_in_ * (self.n_features_in_ + 3) / 2
        neuron_indexes = list(self.neurons_.keys())
        reg_covar = self.reg_covar_ * np.eye(self.n_features_in_)

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
                    self.neurons_[neuron_idx].cov_ = np.dot((responsibilities[:, current_comp][:, np.newaxis] * diff).T,
                                                            diff) / denominator + reg_covar

                    # A part that is able to kill components
                    weights[current_comp] = np.maximum(0, np.sum(responsibilities[:, current_comp]) - N / 2) / X.shape[
                        0]
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
                            times_saved += 1

                    if not was_killed:
                        current_comp += 1

                prev_llikelihood_ = current_llikelihood_
                current_llikelihood_ = self.calculate_ll(X)
                description_length = -current_llikelihood_ + 0.5 * N * np.sum(np.log(weights)) + \
                                     + 0.5 * (N + 1) * k * np.log(X.shape[0])

                delta_llikelihood_ = current_llikelihood_ - prev_llikelihood_
                if np.abs(delta_llikelihood_ / prev_llikelihood_) < self.tol_:
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
