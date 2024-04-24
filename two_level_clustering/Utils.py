import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import root_scalar
import warnings
from typing import Callable, Iterable
from queue import Queue
from collections import OrderedDict
from abc import ABC, abstractmethod

class Neuron:
    def __init__(self, weight, mean, cov):
        self.weight_ = weight
        self.mean_ = mean
        self.cov_ = cov
    
    def __iter__(self):
        yield self.weight_
        yield self.mean_
        yield self.cov_
    
    def copy(self):
        # Create a new instance of Neuron with the same attribute values
        return Neuron(self.weight_, self.mean_, self.cov_)

    def show(self):
        print("Weight:{0}".format(self.weight_))
        print("Mean:{0}".format(self.mean_))
        print("Cov:{0}".format(self.cov_))


class Graph:
    def __init__(self, n, m, graph_type, adj_list=None):
        self.adj_list_ = None
        self.nodes_amount_ = None

        if graph_type == 'rectengular':
            self._create_rectengular(n, m)
        elif adj_list:
            self.adj_list_ = adj_list
            self.nodes_amount_ = len(adj_list)

    def _create_rectengular(self, n, m):
        self.adj_list_ = {}
        idx = lambda r, t: r*m + t 

        self.nodes_amount_ = n*m  

        for i in range(0, n):
            for j in range(0, m):
                # Corner case 
                current_idx = idx(i, j) 

                neighbours = []
                if i - 1 >= 0:
                     neighbours.append(idx(i-1, j))
                if i + 1 < n:
                     neighbours.append(idx(i+1, j))
                if j - 1 >= 0:
                     neighbours.append(idx(i, j-1))
                if j + 1 < m:
                     neighbours.append(idx(i, j+1))
                
                self.adj_list_[current_idx] = neighbours
    

    def delete_vertex(self, v):
        neighbours = self.adj_list_[v]

        for neighbour in neighbours:
            self.adj_list_[neighbour].remove(v)
        
        del self.adj_list_[v]
        self.nodes_amount_ -= 1
    
    def merge_vertices(self, e, v):
        # Merge vertex e and v into vertex e
        for vertex in self.adj_list_[v]:
            if vertex not in self.adj_list_[e] and vertex != e:
                self.create_edge(e, vertex)
        
        self.delete_vertex(v)
    

    def delete_edge(self, e, v):
        self.adj_list_[e].remove(v)
        self.adj_list_[v].remove(e)
    
    def create_edge(self, e, v):
        self.adj_list_[e].append(v)
        self.adj_list_[v].append(e)

    def show(self):
        for v in self.adj_list_:
            print("Vertex->{}".format(v))
            print("Neighbours->", end=' ')
            for e in self.adj_list_[v]:
                print(e, end=' ')
            print()


    def shortest_pairwise_path(self):
        # Calculate all pairwise distances using bfs and return distance matrix
        path_distance = np.zeros(shape=(self.nodes_amount_, self.nodes_amount_))

        vertex_to_idx = {vertex: idx for idx, vertex in enumerate(self.adj_list_)}

        for current_vertex in self.adj_list_:
            parent_vertices = self._bfs(current_vertex)
            for vertex in parent_vertices:
                cur_vertex_idx =  vertex_to_idx[current_vertex]
                vertex_idx = vertex_to_idx[vertex]

                if path_distance[vertex_idx, cur_vertex_idx] != 0:
                    # Symmetric matrix
                    path_distance[cur_vertex_idx, vertex_idx] = path_distance[vertex_idx, cur_vertex_idx]
                    continue

                distance = 0
                prev = vertex
                
                while prev != current_vertex:
                    prev = parent_vertices[prev]
                    distance += 1
                    
                path_distance[cur_vertex_idx, vertex_idx] = distance

        return path_distance
    
    
    def _bfs(self, vertex):
        queue = Queue()
        visited = {key: False for key in self.adj_list_}
        prev = {}

        queue.put(vertex)
        visited[vertex] = True
        
        while not queue.empty():
            node = queue.get()
            neighbours = self.adj_list_[node]

            for neighbour in neighbours:
                if not visited[neighbour]:
                    queue.put(neighbour)
                    visited[neighbour] = True
                    prev[neighbour] = node
        return prev


class Lattice(ABC):
    def __init__(self, size, adj_list=None):
        self.size_ = size
        self.neurons_ = OrderedDict()
        self.neurons_nb_ = None
        self.graph_ = None
        self.pairwise_distance_ = None
        
        if adj_list is not None:
            self.generate(adj_list)
        else:
            self.generate()
    
    def get_weights(self):
        return [neuron.weight_ for neuron in self.neurons_.values()]
    
    def get_means(self):
        return [neuron.mean_ for neuron in self.neurons_.values()]

    def get_covs(self):
        return [neuron.cov_ for neuron in self.neurons_.values()]


    @abstractmethod
    def generate(self):
        pass
    
    def update_distances(self):
        self.pairwise_distance_ = self.graph_.shortest_pairwise_path()
        

    def delete_vertex(self, e):
        self.graph_.delete_vertex(e)

        del self.neurons_[e]
        self.neurons_nb_ -= 1


    def collapse_edge(self, e, v):
        self.graph_.merge_vertices(e, v)
        reestimate_params(self.neurons_[e], self.neurons_[v])

        del self.neurons_[v]
        self.neurons_nb_ -= 1


class RectangularLattice(Lattice):
    def generate(self):
        self.neurons_nb_ = self.size_[0] * self.size_[1]
        self.graph_ = Graph(self.size_[0], self.size_[1], 'rectengular')

        for i in range(0, self.size_[0]):
            for j in range(0, self.size_[1]):
                self.neurons_[i * self.size_[1] + j] = Neuron(None, None, None)

        self.update_distances()


class CustomLattice(Lattice):
    def generate(self, adj_list):
        self.neurons_nb_ = len(adj_list)
        self.graph_ = Graph(None, None, None, adj_list)

        for vertex in adj_list:
            self.neurons_[vertex] = Neuron(None, None, None)
        
        self.update_distances()


def pcws_reg(x, y, verbose=False):
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



def multi_root(f: Callable, bracket: Iterable[float], args: Iterable = (), n: int = 200) -> np.ndarray:
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


def reestimate_params(neuron1, neuron2):
    # Merge neuron1 and neuron2, change neuron1 in place

    weight = neuron1.weight_ + neuron2.weight_

    weight1 = neuron1.weight_ / weight
    weight2 = neuron2.weight_ / weight

    mean = weight1 * neuron1.mean_ + weight2 * neuron2.mean_  
    covariance = weight1 * neuron1.cov_ + weight2 * neuron2.cov_

    neuron1.weight_ = weight
    neuron1.mean_ = mean
    neuron1.cov_ = covariance

def generate_mixture(means, covariances, probabilities, n_samples, random_state=None):
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    mixture_indexes = np.random.choice(a=probabilities.size, p=probabilities, size=n_samples)
    
    labels = []
    data = []
    for idx in mixture_indexes:
        mean = means[idx]
        cov = covariances[idx]

        data.append(np.random.multivariate_normal(mean, cov))
        labels.append(idx)

    return np.array(data), np.array(labels)
