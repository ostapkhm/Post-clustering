from abc import ABC, abstractmethod
from collections import OrderedDict
from queue import Queue
import numpy as np

class Graph:
    def __init__(self, n, m, graph_type):
        self.adj_list_ = None
        self.nodes_amount_ = None

        if graph_type == 'rectengular':
            self._create_rectengular(n, m)


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
            neighbour_list = self.adj_list_[neighbour]
            neighbour_list.remove(v)
        
        del self.adj_list_[v]
        self.nodes_amount_ -= 1
    

    def delete_edge(self, e, v):
        self.adj_list_[e].remove(v)
        self.adj_list_[v].remove(e)
    

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


class Neuron:
    def __init__(self, weight, mean, cov):
        self.weight_ = weight
        self.mean_ = mean
        self.cov_ = cov
    

class Lattice(ABC):
    def __init__(self, size):
        self.size_ = size
        self.neurons_ = OrderedDict()
        self.neurons_nb_ = None
        self.graph_ = None
        self.pairwise_distance_ = None

        self.generate()
    
    @abstractmethod
    def generate(self):
        pass
    
    def update_distances(self):
        self.pairwise_distance_ = self.graph_.shortest_pairwise_path()

    def delete_node(self, v):
        self.graph_.delete_node(v)
        del self.neurons_[v]
        self.neurons_nb_ -= 1 
    
    def delete_edge(self, v1, v2):
        self.graph_.delete_edge(v1, v2)


class RectangularLattice(Lattice):
    def generate(self):
        self.neurons_nb_ = self.size_[0] * self.size_[1]
        self.graph_ = Graph(self.size_[0], self.size_[1], 'rectengular')

        for i in range(0, self.size_[0]):
            for j in range(0, self.size_[1]):
                self.neurons_[i * self.size_[1] + j] = Neuron(None, None, None)

        self.update_distances()
