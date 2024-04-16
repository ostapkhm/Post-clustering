from abc import ABC, abstractmethod
from collections import OrderedDict
from queue import Queue
import numpy as np
import heapq


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

        self.update_distances()


    def collapse_edge(self, e, v):
        self.graph_.merge_vertices(e, v)
        self.reestimate_params(e, v)

        del self.neurons_[v]
        self.neurons_nb_ -= 1

        self.update_distances()
        
    
    def reestimate_params(self, e, v):
        # Reestimate parameters of merged vertex e

        # self.neurons_[e].show()
        # self.neurons_[v].show()

        weight = self.neurons_[e].weight_ + self.neurons_[v].weight_

        weight1 = self.neurons_[e].weight_ / weight
        weight2 = self.neurons_[v].weight_ / weight

        mean = weight1 * self.neurons_[e].mean_ + weight2 * self.neurons_[v].mean_  
        covariance = weight1 * self.neurons_[e].cov_ + weight2 * self.neurons_[v].cov_

        self.neurons_[e].weight_ = weight
        self.neurons_[e].mean_ = mean
        self.neurons_[e].cov_ = covariance

        # self.neurons_[e].show()


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