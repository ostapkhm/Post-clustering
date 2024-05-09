from pbsom.SOM import SOM
import numpy as np
from copy import deepcopy
import networkx as nx
import seaborn as sns
from matplotlib.patches import Ellipse


class Monitor:
    def __init__(self, som:SOM):
        self.model_ = som

        self.idx_ = 0
        self.log_likelihoods = []
        self.vars_ = []
        self.criteria_vals_ = []
        self.labels_ = []

        self.weights_ = []
        self.means_ = []
        self.covs_ = []

        self.neurons_indexes_ = []
        self.neurons_graphs_ = []
    

    def save(self, labels, structure_changed=False, use_criteria=False):
        lattice = self.model_.lattice_

        if structure_changed or self.idx_ == 0:
            self.neurons_indexes_.append(list(lattice.neurons_.keys()))
            self.neurons_graphs_.append(deepcopy(lattice.graph_))
        else:
            # Copying reference to objects
            self.neurons_graphs_.append(self.neurons_graphs_[-1])
            self.neurons_indexes_.append(self.neurons_indexes_[-1])


        self.log_likelihoods.append(self.model_.log_likelihood_)
        self.vars_.append(self.model_.sigma_**2)

        if use_criteria:
            self.criteria_vals_.append([lattice.neurons_nb_, self.model_.criteria_])

        self.weights_.append(lattice.get_weights())
        self.means_.append(lattice.get_means())
        self.covs_.append(lattice.get_covs())

        self.labels_.append(labels)
        self.idx_ += 1
    

    def draw(self, ax, title, epoch_nb, data, means, custom_palette):
        if self.model_.n_features_in_ != 2:
            raise ValueError("Features numbers should be 2!")
        
        artists = []

        # Scatter plot
        scatter = sns.scatterplot(x=data[:, 0], y=data[:, 1], 
                                  alpha=0.7, 
                                  hue=self.labels_[epoch_nb], 
                                  ax=ax, 
                                  palette=custom_palette)
        
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
        for neuron_idx, weight, mean, cov in zip(self.neurons_indexes_[epoch_nb], self.weights_[epoch_nb], self.means_[epoch_nb], self.covs_[epoch_nb]):

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(5.991 * eigenvalues)
            ell = Ellipse(xy=mean, width=width, height=height, angle=angle, color='black', fill=False)
            ax.add_patch(ell)
            ax.text(mean[0], mean[1], str(neuron_idx), color='red', fontsize=12, ha='center', va='center')
            neuron_param = ax.plot(mean[0], mean[1], color='black', markersize=50*weight, marker='*')[0]
            neuron_params.append(neuron_param)

        artists.extend(neuron_params)

        # Draw a grid

        G = nx.DiGraph(self.neurons_graphs_[epoch_nb].adj_list_)

        # Create a Matplotlib figure and axis
        # Draw the graph based on centroids on the specified axis
        centroids = dict(zip(self.neurons_indexes_[epoch_nb], self.means_[epoch_nb]))

        grid = nx.draw_networkx(G, centroids, ax=ax, with_labels=True, node_size=200, 
                         node_color='skyblue', font_size=12, font_weight='bold', arrows=False, alpha=0.6)
    
        artists.append(grid)

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title(title)
        ax.legend(loc='best')

        return artists
