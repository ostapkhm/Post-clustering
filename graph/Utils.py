from abc import ABC, abstractmethod
from collections import OrderedDict
from queue import Queue
import numpy as np
import heapq



class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def __len__(self):
        return len(self._queue)
    


class Neuron:
    def __init__(self, weight, mean, cov):
        self.weight_ = weight
        self.mean_ = mean
        self.cov_ = cov
    
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
        self.max_degree_ = None

        if graph_type == 'rectengular':
            self._create_rectengular(n, m)
        elif adj_list:
            self.adj_list_ = adj_list
            self.nodes_amount_ = len(adj_list)


    def _create_rectengular(self, n, m):
        self.adj_list_ = {}
        idx = lambda r, t: r*m + t 

        self.max_degree_ = 4
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

    def direct_neighbours(self, vertex):
        # Calculate pairwise distances from vertex to it's direct neighbours 
        # using bfs and return distance matrix

        path_distance = np.full((self.nodes_amount_, self.nodes_amount_), np.inf)
        np.fill_diagonal(path_distance, 0)
        
        vertex_to_idx = {vertex: idx for idx, vertex in enumerate(self.adj_list_)}

        
        vertex_idx = vertex_to_idx[vertex]

        for direct_neighbour in self.adj_list_[vertex]:
            neighbour_idx = vertex_to_idx[direct_neighbour]

            path_distance[vertex_idx, neighbour_idx] = 1
            path_distance[neighbour_idx, vertex_idx] = 1
        
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
    
    def update_distances(self, e=None):
        if e is None:
            self.pairwise_distance_ = self.graph_.shortest_pairwise_path()
        else:
            self.pairwise_distance_ = self.graph_.direct_neighbours(e)

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

        self.update_distances(e)
        return

        # Calculate KL divergence for each neighbour of e
        # and remove if their amount exeeds max degree of 
        # vertex in graph

        

        neighbours = np.array(self.graph_.adj_list_[e])
        if len(neighbours) <= self.graph_.max_degree_:
            self.update_distances()
            return
        
        js_divergences = np.zeros(neighbours.shape)
 
        for idx, vertex in enumerate(neighbours):
            js_divergences[idx] = self.__js_divergence(self.neurons_[e], self.neurons_[vertex])
        
        sorted_indices = np.argsort(js_divergences)
    
        # Delete all those edges that are too far from e

        for vertex in neighbours[sorted_indices[self.graph_.max_degree_:]]:
            self.graph_.delete_edge(e, vertex)

        # print('----------------------')
        # self.graph_.show()
        # print('----------------------')
        self.update_distances()

        print("Pairwise distance->", self.pairwise_distance_)


    # def delete_vertex(self, v):
    #     max_degree = 4
    #     for vertex1 in self.graph_.adj_list_[v]:
    #         for vertex2 in self.graph_.adj_list_[v]:
    #             if vertex1 != vertex2:
    #                 # if len(self.graph_.adj_list_[vertex1]) <= max_degree and \
    #                 #     len(self.graph_.adj_list_[vertex2]) <= max_degree: 
    #                 if not(vertex2 in self.graph_.adj_list_[vertex1]):
    #                     self.graph_.create_edge(vertex1, vertex2)

    #     self.graph_.delete_vertex(v)
    #     del self.neurons_[v]
    #     self.neurons_nb_ -= 1

    #     self.update_distances()
    
    def reestimate_params(self, e, v):
        # Reestimate parameters of merged vertex e

        self.neurons_[e].show()
        self.neurons_[v].show()

        weight = self.neurons_[e].weight_ + self.neurons_[v].weight_

        weight1 = self.neurons_[e].weight_ / weight
        weight2 = self.neurons_[v].weight_ / weight

        mean = weight1 * self.neurons_[e].mean_ + weight2 * self.neurons_[v].mean_  

        covariance = weight1 * self.neurons_[e].cov_ + weight2 * self.neurons_[v].cov_ + \
                     weight1 * weight2 * (self.neurons_[e].mean_ - self.neurons_[v].mean_) @ (self.neurons_[e].mean_ - self.neurons_[v].mean_).T

        self.neurons_[e].weight_ = weight
        self.neurons_[e].mean_ = mean
        self.neurons_[e].cov_ = covariance

        self.neurons_[e].show()


    def reconstruct_map(self):
        heaps = [PriorityQueue() for _ in range(self.neurons_nb_)]

        to_idx = {}

        for i, (idx1, neuron1) in enumerate(self.neurons_.items()):
            to_idx[idx1] = i
            for j, (idx2, neuron2) in enumerate(self.neurons_.items()):
                if i != j:
                    heaps[i].push((min(idx1, idx2), max(idx1, idx2)), self.__euclidean_distance(neuron1, neuron2))
                    

        ### Erease adj_list_
        self.graph_.adj_list_ = {key: [] for key in self.graph_.adj_list_}

        
        max_neighbours_nb = min(self.neurons_nb_ - 1, self.graph_.max_degree_)
        occurrences = np.zeros(self.neurons_nb_ * self.neurons_nb_)
        for _ in range(max_neighbours_nb):
            for i in range(self.neurons_nb_):
                u, v = heaps[i].pop()
                u_idx, v_idx = to_idx[u], to_idx[v]
                
                idx = u_idx * self.neurons_nb_ + v_idx
                occurrences[idx] += 1

                if occurrences[idx] == 2:
                    self.graph_.adj_list_[u].append(v)
                    self.graph_.adj_list_[v].append(u)
        
        
        print("Neighbours updated!")
        self.graph_.show()

    
    def __js_divergence(self, neuron1, neuron2):
        return 0.5 * (self.__kl_divergence(neuron1, neuron2) + self.__kl_divergence(neuron2, neuron1))


    @staticmethod
    def __kl_divergence(neuron1:Neuron, neuron2:Neuron):
        k = len(neuron1.mean_)  # Dimensionality of the distributions
        cov_q_inv = np.linalg.inv(neuron2.cov_)
        
        kl_divergence = 0.5 * (
            np.trace(np.dot(cov_q_inv, neuron1.cov_)) +
            np.dot(np.dot((neuron2.mean_- neuron1.mean_).T, cov_q_inv), (neuron2.mean_- neuron1.mean_)) -
            k + 
            np.log(np.linalg.det(neuron2.cov_) / np.linalg.det(neuron1.cov_))
        )

        return kl_divergence

    @staticmethod
    def __euclidean_distance(neuron1:Neuron, neuron2:Neuron):
        return np.sqrt(np.sum(np.square(neuron1.mean_ - neuron2.mean_)))



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