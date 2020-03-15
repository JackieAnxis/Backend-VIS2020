import numpy as np
from scipy import sparse

def third_node(graph):
    id2index = {}
    nodes = np.array([[data['x'], data['y']]
                      for (id, data) in graph.nodes.data()])  # position
    i = 0
    for node in graph.nodes.data():
        id = node[0]
        id2index[id] = i
        i += 1
    links = []
    for link in graph.edges.data():
        links.append([id2index[link[0]], id2index[link[1]]])
    links = np.array(links)

    nodes_0_index = links[:, 0]  # id
    nodes_1_index = links[:, 1]  # id
    vectors = nodes[nodes_1_index, :] - nodes[nodes_0_index, :]
    perpendicular_vectors = np.stack([-vectors[:, 1], vectors[:, 0]])
    perpendicular_vectors = (perpendicular_vectors / np.sqrt(np.sum(perpendicular_vectors**2, axis=0))).T
    nodes_0_pos = nodes[nodes_0_index, :]
    nodes_2_pos = nodes_0_pos + perpendicular_vectors
    new_nodes = np.vstack((nodes, nodes_2_pos))
    new_nodes_index = nodes.shape[0] + np.array(range(0, links.shape[0]))
    new_links = np.hstack((links, new_nodes_index[:, np.newaxis]))
    V = []
    for i in range(0, new_links.shape[0]):
        V.append(
            np.transpose(
                np.vstack((new_nodes[new_links[i, 1], :] - new_nodes[new_links[i, 0], :],
                           new_nodes[new_links[i, 2], :] - new_nodes[new_links[i, 0], :]))))
    # V =
    # [n1.x-n0.x, n2.x-n0.x]
    # [n1.y-n0.y, n2.y-n0.y]
    return V, perpendicular_vectors, new_nodes, new_links, id2index

def get_link_directions(graph):
    id2index = {}
    nodes = np.array([[data['x'], data['y']]
                      for (id, data) in graph.nodes.data()])  # position
    i = 0
    for node in graph.nodes.data():
        id = node[0]
        id2index[id] = i
        i += 1
    links = []
    for link in graph.edges.data():
        links.append(nodes[id2index[link[0]]] - nodes[id2index[link[1]]])
    links = np.array(links)

    links_length = np.sqrt(np.sum(links**2, axis=1))
    links_length[np.where(links_length == 0)] = 1
    links[:, 0] = links[:, 0] / links_length
    links[:, 1] = links[:, 1] / links_length
    return links

def calculate_T(source_graph, target_graph, correspondences):
    correspondence_count = 0
    for correspondence in correspondences:
        if correspondence.shape[0] > 0:
            correspondence_count += correspondence.shape[0]
        else:
            correspondence_count += 1

    source_links = get_link_directions(source_graph)
    target_links = get_link_directions(target_graph)

    # X * T = _X
    X = np.zeros((correspondence_count * 3, 3 * 3))
    _X = np.zeros((correspondence_count * 3, 1))
    offset = 0
    for i in range(0, len(correspondences)):
        correspondence = correspondences[i]
        target_link = target_links[i]
        x = target_link[0]
        y = target_link[1]
        if correspondence.shape[0] > 0:
            for j in range(0, correspondence.shape[0]):
                source_link = source_links[correspondence[j]]
                _x = source_link[0]
                _y = source_link[1]
                X[offset, :] = np.array([x, y, 1, 0, 0, 0, 0, 0, 0])
                X[offset+1, :] = np.array([0, 0, 0, x, y, 1, 0, 0, 0])
                X[offset+2, :] = np.array([0, 0, 0, 0, 0, 0, x, y, 1])
                _X[offset] = _x
                _X[offset+1] = _y
                _X[offset+2] = 1
                offset += 3
    x = sparse.linalg.lsqr(X, _X, iter_lim=5000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
    T = np.vstack([np.array([x[0][0], x[0][1], x[0][2]]), np.array([x[0][3], x[0][4], x[0][5]]), np.array([x[0][6], x[0][7], x[0][8]])])
    
    return T

def rotate(graph, T):
    rotated_graph = graph.copy()
    i = 0
    for node in rotated_graph.nodes.data():
        node = rotated_graph.nodes[i]
        v = np.array([node['x'], node['y'], 1])
        node['x'] = np.dot(T[0, :], v)
        node['y'] = np.dot(T[1, :], v)
        i += 1
    return rotated_graph