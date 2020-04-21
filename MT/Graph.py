import scipy.sparse
import numpy as np
import networkx as nx

def straddle(edge_0, edge_1):
    # v1 = another_segment.point1 - self.point1  
    v1 = edge_1[0] - edge_0[0]
    v2 = edge_1[1] - edge_0[0]
    vm = edge_0[1] - edge_0[0]
    if np.cross(v1, vm) * np.cross(v2, vm) <= 0:
        return True  
    else:  
        return False
        
def is_cross(edge_0, edge_1):
    if straddle(edge_0, edge_1) and straddle(edge_1, edge_0):  
        return True  
    else:  
        return False  

class Graph():
    def __init__(self, graph):
        graph = nx.Graph(graph)
        str_map = {}
        for node in graph.nodes:
            str_map[node] = str(node)
        graph = nx.relabel_nodes(graph, str_map)
        id2index = {}
        index2id = {}
        graph_nodes_data = sorted(graph.nodes.data(), key=lambda x: x[0])
        nodes = np.array([[data['x'], data['y']]
                      for (id, data) in graph_nodes_data])  # position
        i = 0
        for node in graph_nodes_data:
            id = str(node[0])
            id2index[id] = i
            index2id[i] = id
            i += 1
        graph_edges_data = sorted(graph.edges.data(), key=lambda x: x[0]+x[1])
        edges = []
        for edge in graph_edges_data:
            edges.append([id2index[str(edge[0])], id2index[str(edge[1])]])
        edges = np.array(edges, dtype=np.int32)
        
        self.nodes = nodes
        self.edges = edges
        self.weights = np.ones((edges.shape[0], 1))
        self.id2index = id2index
        self.index2id = index2id
        self.rawgraph = graph
        self.adj_matrix = np.zeros((0, 0))
        self.euc_adj_matrix = np.zeros((0, 0))
        self.euc_adj_R = 0
        self.graph_distance_matrix = np.zeros((0, 0))
    
    def copy(self):
        new_G = Graph(self.to_networkx())
        new_G.graph_distance_matrix = self.graph_distance_matrix.copy()
        return new_G

    def to_networkx(self):
        # graph = nx.Graph()
        # for edge in self.new_edges:
        #     graph.add_edge(str(edge[0]), str(edge[1]))
        #     graph.add_edge(str(edge[0]), str(edge[2]))
        # index = 0
        # for node in self.new_nodes:
        #     id = str(index)
        #     graph.nodes[id]['x'] = node[0]
        #     graph.nodes[id]['y'] = node[1]
        #     index += 1
        # return graph
        graph = self.rawgraph.copy()
        for node in graph.nodes.data():
            id = node[0]
            index = self.id2index[str(id)]
            pos = self.nodes[index]
            graph.nodes[id]['x'] = pos[0]
            graph.nodes[id]['y'] = pos[1]
        return graph

    def normalize(self, std=np.sqrt(2), ctr=0):
        nodes_count = self.nodes.shape[0]
        dimension = 2
        center = np.mean(self.nodes, axis=0)
        mean_distance2center = np.mean(np.sqrt(np.sum((self.nodes-center)**2, axis=1)), axis=0)
        scale = std / mean_distance2center
        # T = np.eye(dimension + 1)
        # T[0:dimension, dimension] = (ctr - center.T)
        # nodes = np.concatenate((self.nodes, np.ones((nodes_count, 1))), axis=1)
        # nodes = scale*nodes.dot(T.T)
        # self.nodes = nodes[:, 0:dimension]
        T = np.eye(dimension) * scale
        nodes = self.nodes.dot(T.T)
        nodes += ctr - np.mean(nodes, axis=0)
        self.nodes = nodes[:, 0:dimension]
        for id in self.rawgraph.nodes:
            self.rawgraph.nodes[id]['x'] = self.nodes[self.id2index[str(id)]][0]
            self.rawgraph.nodes[id]['y'] = self.nodes[self.id2index[str(id)]][1]

    def compute_third_node(self):
        nodes = self.nodes
        nodes_0_index = self.edges[:, 0]  # id
        nodes_1_index = self.edges[:, 1]  # id
        vectors = nodes[nodes_1_index, :] - nodes[nodes_0_index, :]
        # Rotate 90 degrees counterclockwise
        perpendicular_vectors = np.stack([-vectors[:, 1], vectors[:, 0]])
        perpendicular_vectors = (perpendicular_vectors / np.sqrt(np.sum(perpendicular_vectors**2, axis=0))).T
        
        nodes_0_pos = nodes[nodes_0_index, :]
        nodes_2_pos = nodes_0_pos + perpendicular_vectors
        
        max_node_id = np.max([int(id) for (id, data) in self.rawgraph.nodes.data()])
        i = 0
        for node in nodes_2_pos:
            max_node_id += 1
            id = max_node_id
            index = self.nodes.shape[0] + i
            self.id2index[id] = index
            self.index2id[index] = id
            i += 1
        
        new_nodes = np.vstack((nodes, nodes_2_pos))
        new_nodes_index = nodes.shape[0] + np.array(range(0, self.edges.shape[0]))
        new_edges = np.hstack((self.edges, new_nodes_index[:, np.newaxis]))
        V = []
        # V = [n1-n0, n2-n0]
        # [n1.x-n0.x, n2.x-n0.x]
        # [n1.y-n0.y, n2.y-n0.y]
        for i in range(0, new_edges.shape[0]):
            Vi = np.vstack((new_nodes[new_edges[i, 1], :] - new_nodes[new_edges[i, 0], :],
                            new_nodes[new_edges[i, 2], :] - new_nodes[new_edges[i, 0], :]))
            V.append(Vi.T)
        self.V = V
        self.new_nodes = new_nodes
        self.new_edges = new_edges
        self.perpendicular = perpendicular_vectors
    
    def find_adj_edges(self):
        adj_matrix = np.zeros((self.edges.shape[0], self.edges.shape[0]))
        for i in range(0, self.edges.shape[0]):
            edge_0 = self.edges[i]
            for j in range(i + 1, self.edges.shape[0]):
                edge_1 = self.edges[j]
                # whether they use the same node
                if edge_0[0] == edge_1[0] or edge_0[1] == edge_1[0] or edge_0[0] == edge_1[1] or edge_0[1] == edge_1[1]:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1
                elif is_cross(self.nodes[edge_0, :], self.nodes[edge_1, :]):
                    adj_matrix[i, j] = adj_matrix[j, i] = 1
                else:
                    adj_matrix[i, j] = adj_matrix[j, i] = 0
        return adj_matrix

    def build_elementary_cell(self):
        '''
        U is the inverse of V, this function aims to computer the A matrix:
        [-(u00+u10) u00 u10]
        [-(u01+u11) u01 u11]
        '''
        edges_count = self.edges.shape[0]
        V = self.V
        A = [0 for i in range(0, edges_count)]
        for i in range(0, edges_count):
            U = np.linalg.inv(V[i])
            A[i] = np.hstack((-np.sum(U, axis=0).T[:, np.newaxis], U.T))
        self.A = A
        return A
    
    def compute_edges_center(self):
        self.edges_center = np.zeros((self.edges.shape[0], 2))
        for i in range(0, self.edges.shape[0]):
            self.edges_center[i, :] = np.mean(self.nodes[self.edges[i], :], axis=0)
        return self.edges_center

    def compute_adjacent_matrix(self):
        if self.adj_matrix.shape[0]:
            return self.adj_matrix
        self.adj_matrix = np.zeros((self.nodes.shape[0], self.nodes.shape[0]))
        for i in range(0, self.edges.shape[0]):
            node_0 = self.edges[i][0]
            node_1 = self.edges[i][1]
            self.adj_matrix[node_0, node_1] = self.weights[i]
            self.adj_matrix[node_1, node_0] = self.weights[i]
        return self.adj_matrix
    
    def rw_laplacian_matrix(self, adj_matrix):
        # Laplacian matrix: D-A
        adj = adj_matrix
        deg = np.diag(np.sum(adj, axis=0))
        lap = deg - adj
        # lap = nx.laplacian_matrix(g)
        # adj = nx.adjacency_matrix(g)
        inv_deg = scipy.sparse.diags(1/adj.dot(np.ones([adj.shape[0]])))
        return inv_deg.dot(lap)

    def laplacian_matrix(self, adj_matrix):
        # Laplacian matrix: D-A
        adj = adj_matrix
        deg = np.diag(np.sum(adj, axis=0))
        lap = deg - adj
        return lap

    def get_local_neighbor(self, i, adj_matrix):
        nb = []
        for j in range(0, adj_matrix.shape[0]):
            if adj_matrix[i][j]:
                nb.append(j)
        return nb

    def compute_euc_adj_matrix(self, R):
        if self.euc_adj_matrix.shape[0] and self.euc_adj_R == R:
            return self.euc_adj_matrix
        else:
            self.compute_adjacent_matrix()
            self.euc_adj_matrix = np.zeros((self.nodes.shape[0], self.nodes.shape[0]))
            for i in range(0, self.nodes.shape[0]):
                node_0 = self.nodes[i]
                for j in range(i + 1, self.nodes.shape[0]):
                    node_1 = self.nodes[j]
                    distance = np.sqrt(np.sum((node_1-node_0)**2))
                    if distance <= R or (self.adj_matrix[i, j]):
                    # if distance <= R or (self.adj_matrix[i, j] and distance <= 1.5 * R):
                        self.euc_adj_matrix[i, j] = 1
                        self.euc_adj_matrix[j, i] = 1
            return self.euc_adj_matrix

    def compute_graph_distance_matrix(self):
        if self.graph_distance_matrix.shape[0]:
            return self.graph_distance_matrix
        distances = dict(nx.all_pairs_shortest_path_length(self.rawgraph))
        self.graph_distance_matrix = np.zeros((self.nodes.shape[0], self.nodes.shape[0]))
        for i in range(self.nodes.shape[0]):
            idi = self.index2id[i]
            for j in range(i + 1, self.nodes.shape[0]):
                idj = self.index2id[j]
                self.graph_distance_matrix[i, j] = self.graph_distance_matrix[j, i] = distances[idi][idj]
        return self.graph_distance_matrix



