from pbsom.SOM import SOM
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse


class Monitor:
    def __init__(self, som:SOM):
        self.lattice_ = som.lattice_
        self.model_ = som
    
    def initialize_params(self):
        epochs = self.model_.max_iter_
        features_nb = self.model_.n_features_in_
    
        self.log_likelihoods = np.zeros(epochs)
        self.vars_ = np.zeros(epochs)
        self.weights_ = np.zeros(shape=(epochs, self.lattice_.neurons_nb_))
        self.means_ = np.zeros(shape=(epochs, self.lattice_.neurons_nb_, features_nb))
        self.covs_ = np.zeros(shape=(epochs, self.lattice_.neurons_nb_, features_nb, features_nb))

        self.idx_ = 0

    def save(self):
        self.log_likelihoods[self.idx_] = self.model_.log_likelihood
        self.vars_[self.idx_] =  self.model_.sigma_**2
        self.weights_[self.idx_] = np.array([neuron.weight_ for neuron in self.lattice_.neurons_])
        self.means_[self.idx_] = np.array([neuron.mean_ for neuron in self.lattice_.neurons_])
        self.covs_[self.idx_] = np.array([neuron.cov_ for neuron in self.lattice_.neurons_])
        self.idx_ += 1
    

    def draw(self, ax, title, epoch_nb, data, means, custom_palette, labels):
        if self.model_.n_features_in_ != 2:
            raise ValueError("Features numbers should be 2!")
        
        artists = []

        # Scatter plot
        if labels is not None:
            scatter = sns.scatterplot(x=data[:, 0], y=data[:, 1], alpha=0.7, hue=labels, ax=ax, palette=custom_palette)
        else:
            scatter = sns.scatterplot(x=data[:, 0], y=data[:, 1], alpha=0.7, ax=ax)
        
        artists.append(scatter)

        # Plot true parameters
        if means is not None:
            true_params = []
            for mean, color in zip(means, custom_palette):
                true_param = ax.plot(mean[0], mean[1], color=color, markersize=12, marker='^')[0]
                true_params.append(true_param)
            artists.extend(true_params)

        # Plot neuron parameters
        neuron_params = []
        for weight, mean, cov in zip(self.weights_[epoch_nb], self.means_[epoch_nb], self.covs_[epoch_nb]):
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(5.991 * eigenvalues)
            ell = Ellipse(xy=mean, width=width, height=height, angle=angle, color='black', fill=False)
            ax.add_patch(ell)
            neuron_param = ax.plot(mean[0], mean[1], color='black', markersize=50*weight, marker='*')[0]
            neuron_params.append(neuron_param)
        artists.extend(neuron_params)

        # Draw a grid
        means = np.vstack((self.means_[epoch_nb], self.means_[epoch_nb][0]))
        grid = ax.plot(means[:, 0], means[:, 1], 'k-')[0]
        artists.append(grid)

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title(title)
        ax.legend(loc='best')

        return artists
