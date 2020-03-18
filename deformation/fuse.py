import networkx as nx
import time
import json
import math
import numpy as np
from scipy import sparse
from networkx.readwrite import json_graph
from scipy.optimize import least_squares, newton, minimize
# from Graph import Graph
# from utils import load_json_graph, save_json_graph
from deformation.Graph import Graph
from deformation.utils import load_json_graph, save_json_graph

def cal_radius(G):
    r = np.max(np.sqrt(np.sum((G.nodes[G.edges[:, 0]] - G.nodes[G.edges[:, 1]]) ** 2, axis=1)), axis=0)
    return r

def target_length_weight(G, subgraphs, need_scaled, weights):
    adj = G.euc_adj_matrix
    _adj = G.adj_matrix
    all_fused_nodes = {}
    fused_nodes_belong_to = {} # the node belong to witch subgraph
    all_fused_nodes_weights = {}
    _G = G.copy()
    for i in range(0, len(subgraphs)):
        subgraph = subgraphs[i]
        fused_nodes_id = np.array([G.id2index[id] for id in subgraph.id2index.keys()])

        ############ scale the subgraph to the original layout ############
        if need_scaled[i]:
            subgraph = scale(G, subgraph)

        for id in subgraph.id2index.keys():
            position = subgraph.nodes[subgraph.id2index[id]]
            if id not in all_fused_nodes:
                all_fused_nodes[id] = []
                all_fused_nodes_weights[id] = []
                fused_nodes_belong_to[id] = []
            all_fused_nodes_weights[id].append(weights[i])
            fused_nodes_belong_to[id].append(i)
            all_fused_nodes[id].append(position)

        for id in subgraph.id2index:
            _G.nodes[_G.id2index[id]] = subgraph.nodes[subgraph.id2index[id]]

    for id in all_fused_nodes:
        all_fused_nodes[id] = np.mean(all_fused_nodes[id], axis=0)
        all_fused_nodes_weights[id] = np.mean(all_fused_nodes_weights[id])

    mean_weight = np.mean(weights, dtype=np.float32)
    coefficients = [1.0/80, 1.0/150, 1.0/150]
    # w0 = 1
    # w1 = 2
    # w2 = 5
    # w3 = 10
    n = G.nodes.shape[0]
    s = int(n * (n - 1) / 2)
    M = np.zeros((s * 2, n * 2))
    C = np.zeros((s * 2, 1))
    offset = 0
    count = [0,0,0,0,0]
    for i in range(0, n):
        node_0_id = G.index2id[i]
        for j in range(i + 1, n):
            node_1_id = G.index2id[j]
            subgraphs_belong_to = np.zeros((0,0))
            if node_0_id in fused_nodes_belong_to and node_1_id in fused_nodes_belong_to:
                subgraphs_belong_to = np.intersect1d(fused_nodes_belong_to[node_0_id], fused_nodes_belong_to[node_1_id])
            if subgraphs_belong_to.shape[0]:
                count[0] += 1
                # two nodes are in one subgraph, with the subgraph's weight
                for k in subgraphs_belong_to:
                    subgraph = subgraphs[k]
                    weight = np.mean(weights[k])
                    M[offset, i * 2] += weight # x
                    M[offset + 1, i * 2 + 1] += weight # y
                    M[offset, j * 2] += -weight # x
                    M[offset + 1, j * 2 + 1] += -weight # y
                    # target_d = G.nodes[i] - G.nodes[j]
                    target_d = subgraph.nodes[subgraph.id2index[node_0_id]] - subgraph.nodes[subgraph.id2index[node_1_id]]
                    if (np.sum(np.isnan(target_d[:, np.newaxis] * weight))):
                        print('overlap!', i, j)
                    C[offset: offset + 2] += target_d[:, np.newaxis] * weight
            elif (node_0_id in all_fused_nodes or node_1_id in all_fused_nodes) and _adj[i,j]:
                # one or two nodes are the fused node, and the node pair is an edge
                count[1] += 1
                target_d = G.nodes[i] - G.nodes[j]
                weight = mean_weight * coefficients[0] / np.sqrt(np.sum(target_d ** 2))
                M[offset, i * 2] = weight  # x
                M[offset + 1, i * 2 + 1] = weight  # y
                M[offset, j * 2] = -weight  # x
                M[offset + 1, j * 2 + 1] = -weight  # y
                if (np.sum(np.isnan(target_d[:, np.newaxis] * weight))):
                    print('overlap!', i, j)
                C[offset: offset + 2] = target_d[:, np.newaxis] * weight
            elif (node_0_id in all_fused_nodes or node_1_id in all_fused_nodes) and adj[i,j]:
                # one or two nodes are the fused node, and the node pair is close in layout
                count[2] += 1
                target_d = G.nodes[i] - G.nodes[j]
                weight = mean_weight * coefficients[1] / np.sqrt(np.sum(target_d ** 2))
                M[offset, i * 2] = weight  # x
                M[offset + 1, i * 2 + 1] = weight  # y
                M[offset, j * 2] = -weight  # x
                M[offset + 1, j * 2 + 1] = -weight  # y
                if (np.sum(np.isnan(target_d[:, np.newaxis] * weight))):
                    print('overlap!', i, j)
                C[offset: offset + 2] = target_d[:, np.newaxis] * weight
            else:
                count[4] += 1
                target_d = G.nodes[i] - G.nodes[j]
                weight = mean_weight * coefficients[2] / np.sqrt(np.sum(target_d ** 2))
                M[offset, i * 2] = weight  # x
                M[offset + 1, i * 2 + 1] = weight  # y
                M[offset, j * 2] = -weight  # x
                M[offset + 1, j * 2 + 1] = -weight  # y
                if (np.sum(np.isnan(target_d[:, np.newaxis] * weight))):
                    print('overlap!', i, j)
                C[offset: offset + 2] = target_d[:, np.newaxis] * weight
            
            offset += 2
    print(count)
    return M, C, _G

def scale(G, subgraph):
    fused_nodes_id = np.zeros((subgraph.nodes.shape[0], 1), dtype=np.int32)
    for i in range(0, fused_nodes_id.shape[0]):
        fused_nodes_id[i] = G.id2index[subgraph.index2id[i]]
    origin_cen = np.mean(G.nodes[fused_nodes_id], axis=0)
    cen = np.mean(subgraph.nodes, axis=0)
    mean_edge_length = np.min(np.sqrt(np.sum((subgraph.nodes[subgraph.edges[:, 0]] - subgraph.nodes[subgraph.edges[:, 1]]) ** 2, axis=1)), axis=0)
    raw_edges = G.edges[(np.isin(G.edges[:, 0], fused_nodes_id) + np.isin(G.edges[:, 1], fused_nodes_id))]
    raw_mean_edge_length = np.min(np.sqrt(np.sum((G.nodes[raw_edges[:, 0]] - G.nodes[raw_edges[:, 1]]) ** 2, axis=1)), axis=0)

    mean_distance2center = np.mean(np.sqrt(np.sum((subgraph.nodes - cen) ** 2, axis=1)), axis=0) * raw_mean_edge_length / mean_edge_length
    subgraph.normalize(mean_distance2center, origin_cen)
    return subgraph

def fuse(G, subgraphs, need_scaled, weights):
    G = G.copy()
    # using the mean length of the edge in G to build the distance adjacent matrix
    # radius = np.mean(np.sqrt(np.sum((G.nodes[G.edges[:, 0]]-G.nodes[G.edges[:, 1]])**2, axis=1)), axis=0)
    radius = cal_radius(G)
    adj = G.compute_euc_adj_matrix(radius)

    ############# Laplacian editing #############
    L = G.rw_laplacian_matrix(adj)
    V = G.nodes
    Delta = L.dot(V)

    n = G.nodes.shape[0]
    index = np.arange(L.shape[0], dtype=np.int32)
    LS = np.zeros([2 * n, 2 * n])
    for i in index:
        LS[i * 2, index * 2] = (-1) * L[i]
        LS[i * 2 + 1, index * 2 + 1] = (-1) * L[i]

    C = np.zeros((n * 2 * 2, 1))

    for i in range(L.shape[0]):
        nb_idx = G.get_local_neighbor(i, adj)  # node i and its neibors
        ring = np.array([i] + nb_idx)  # node i and its neibors in subgraph
        V_ring = V[ring]
        n_ring = V_ring.shape[0]

        A = np.zeros([n_ring * 2, 4])
        for j in range(n_ring):
            A[j] = [V_ring[j, 0], -V_ring[j, 1], 1, 0]
            A[j + n_ring] = [V_ring[j, 1], V_ring[j, 0], 0, 1]

        # Moore-Penrose Inversion
        A_pinv = np.linalg.pinv(A)
        # A_pinv = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A))
        s = A_pinv[0]
        h = A_pinv[1]

        T_delta = np.vstack([
            Delta[i, 0] * s - Delta[i, 1] * h,
            Delta[i, 0] * h + Delta[i, 1] * s
        ])

        LS[i * 2, np.hstack([ring * 2, ring * 2 + 1])] += T_delta[0]
        LS[i * 2 + 1, np.hstack([ring * 2, ring * 2 + 1])] += T_delta[1]

    all_fused_nodes = {}
    all_fused_nodes_weights = {}
    _G = G.copy()
    for i in range(0, len(subgraphs)):
        subgraph = subgraphs[i]
        fused_nodes_id = np.array([G.id2index[id] for id in subgraph.id2index.keys()])

        ############ scale the subgraph to the original layout ############
        if need_scaled[i]:
            subgraph = scale(G, subgraph)

        for id in subgraph.id2index:
            _G.nodes[_G.id2index[id]] = subgraph.nodes[subgraph.id2index[id]]

        for id in subgraph.id2index.keys():
            position = subgraph.nodes[subgraph.id2index[id]]
            if id not in all_fused_nodes:
                all_fused_nodes[id] = []
                all_fused_nodes_weights[id] = []
            all_fused_nodes_weights[id].append(weights[i])
            all_fused_nodes[id].append(position)

    # save_json_graph(_G.to_networkx(), prefix + 'new.json')
    for id in all_fused_nodes:
        all_fused_nodes[id] = np.mean(all_fused_nodes[id], axis=0)
        all_fused_nodes_weights[id] = np.mean(all_fused_nodes_weights[id])
    constraint_coef = np.zeros((n * 2, n * 2))
    for i in range(0, G.nodes.shape[0]):
        row = n * 2 + i * 2
        id = G.index2id[i]
        # fixed nodes
        if id not in all_fused_nodes:
            constraint_coef[i * 2] = (np.arange(2 * n) == i * 2)
            constraint_coef[i * 2 + 1] = (np.arange(2 * n) == i * 2 + 1)
            C[row:row + 2, :] = G.nodes[i][:, np.newaxis]
        if id in all_fused_nodes:
            constraint_coef[i * 2] = (np.arange(2 * n) == i * 2) * all_fused_nodes_weights[id]
            constraint_coef[i * 2 + 1] = (np.arange(2 * n) == i * 2 + 1) * all_fused_nodes_weights[id]
            fused_node_id = G.index2id[i]
            C[row:row + 2, :] = all_fused_nodes[fused_node_id][:, np.newaxis] * all_fused_nodes_weights[id]
    A = np.vstack([LS, constraint_coef])
    M = sparse.coo_matrix(A)
    X = sparse.linalg.lsqr(M, C, iter_lim=30000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
    G.nodes = X[0].reshape((G.nodes.shape[0], 2))
    return G, _G

def fuse_v2(G, subgraphs, need_scaled, weights):
    G = G.copy()
    G.normalize()
    print('Building target length...')
    # using the mean length of the edge in G to build the distance adjacent matrix
    # radius = np.mean(np.sqrt(np.sum((G.nodes[G.edges[:, 0]] - G.nodes[G.edges[:, 1]]) ** 2, axis=1)), axis=0)
    radius = cal_radius(G)
    G.compute_euc_adj_matrix(radius)
    G.compute_adjacent_matrix()
    
    M, C, _G = target_length_weight(G, subgraphs, need_scaled, weights)

    # M = sparse.coo_matrix(M)
    print('begin to fuse...')
    X = sparse.linalg.lsqr(M, C, iter_lim=30000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
    G.nodes = X[0].reshape((G.nodes.shape[0], 2))

    return G, _G

def fuse_v4(G, subgraphs, need_scaled, weights):
    G = G.copy()
    # G.normalize()
    radius = cal_radius(G)
    
    fused_nodes_dict = {}
    for i in range(0, len(subgraphs)):
        subgraph = subgraphs[i]
        for id in subgraph.id2index:
            index = G.id2index[id]
            if index not in fused_nodes_dict:
                fused_nodes_dict[index] = []
            fused_nodes_dict[index].append(subgraph.nodes[subgraph.id2index[id]])
    fused_nodes = list(fused_nodes_dict.keys())
    _adj = G.compute_adjacent_matrix()
    surr_nodes = {}
    for i in G.index2id:
        for j in fused_nodes:
            if i not in fused_nodes:
                dis = np.sqrt(np.sum((G.nodes[i] - G.nodes[j])**2))
                if dis <= radius or _adj[i, j]:
                    surr_nodes[i] = 1
    surr_nodes = list(surr_nodes.keys())

    surr_nodes = surr_nodes + fused_nodes
    print([G.index2id[i] for i in surr_nodes])
    surr_graph = G.rawgraph.subgraph([G.index2id[i] for i in surr_nodes])
    # save_json_graph(surr_graph, './data/bn-mouse-kasthuri/surr_graph.json')
    surr_G = Graph(surr_graph)

    r = cal_radius(surr_G)
    surr_G.compute_euc_adj_matrix(r)
    surr_G.compute_adjacent_matrix()

    M, C, _surr_G = target_length_weight(surr_G, subgraphs, need_scaled, weights)

    ##### fix the nodes into their position #####
    _M = np.eye(surr_G.nodes.shape[0] * 2, surr_G.nodes.shape[0] * 2)
    _C = surr_G.nodes.flatten()
    for i in surr_G.index2id:
        id = surr_G.index2id[i]
        if id in fused_nodes:
            _C[i * 2:i * 2 + 1] = np.mean(fused_nodes_dict[id], axis=0)[:, np.newaxis]
    _C = _C.reshape((_C.shape[0], 1))
    M = np.vstack((M, _M))
    C = np.vstack((C, _C))
    # M = sparse.coo_matrix(M)
    print('begin to fuse...')
    X = sparse.linalg.lsqr(M, C, iter_lim=30000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
    surr_G.nodes = X[0].reshape((int(X[0].shape[0] / 2), 2))
    _G = G.copy()
    for i in surr_G.index2id:
        id = surr_G.index2id[i]
        G.nodes[G.id2index[id]] = surr_G.nodes[i]
        _G.nodes[_G.id2index[id]] = _surr_G.nodes[i]

    return G, _G

def fuse_v3(G, subgraphs, need_scaled, weights):
    G = G.copy()
    G.normalize()
    # using the mean length of the edge in G to build the distance adjacent matrix
    # radius = np.mean(np.sqrt(np.sum((G.nodes[G.edges[:, 0]] - G.nodes[G.edges[:, 1]]) ** 2, axis=1)), axis=0)
    radius = cal_radius(G)
    G.compute_euc_adj_matrix(radius)
    G.compute_adjacent_matrix()

    M, C, _G = target_length_weight(G, subgraphs, need_scaled, weights)

    x0 = M.dot(_G.nodes.flatten())
    pair_index = []
    for i in range(0, G.nodes.shape[0]):
        for j in range(i + 1, G.nodes.shape[0]):
            pair_index.append([i, j])
    pair_index = np.array(pair_index, dtype=np.int32)
    pair_expected_length = (np.sqrt(np.sum(C.reshape((int(C.shape[0] / 2), 2)) ** 2, axis=1)))

    # def resSimXform(x):
    #     x = x.reshape(((int(x.shape[0] / 2), 2)))
    #     return np.sum(((np.sqrt(np.sum((x[pair_index[:, 0]] - x[pair_index[:, 1]])**2, axis=1))) - pair_expected_length)**2)
    #
    # b = newton(resSimXform, x0)
    def resSimXform(x, A, B):
        x = x.reshape(((int(x.shape[0] / 2), 2)))
        return np.sum(((np.sqrt(np.sum((x[A[:, 0]] - x[A[:, 1]])**2, axis=1))) - B)**2)
    b = least_squares(fun=resSimXform, x0=x0, jac='2-point', method='trf', args=(pair_index, pair_expected_length),
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)

    G.nodes = b.x.reshape(((int(b.x.shape[0] / 2), 2)))
    return G, _G


def fuse_main(G, subgraphs):
    need_scaled = []
    weights = []
    for i in range(0, len(subgraphs)):
      subgraphs[i] = Graph(subgraphs[i])
      need_scaled.append(False)
      weights.append(15)
    time_start = time.time()
    new_G,_new_G = fuse_v4(Graph(G), subgraphs, need_scaled, weights)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    return new_G.to_networkx()

def generate_subgraph(prefix):
    with open(prefix + 'graph-pos.json') as f:
        js_graph = json.load(f)
    G = json_graph.node_link_graph(js_graph)

    ring = [61, 59, 65, 63, 60, 66, 76, 64, 62]

    ring = [55]
    for i in nx.neighbors(G, ring[0]):
        if i not in ring:
            ring.append(i)
        # for j in nx.neighbors(G, i):
        #     if j not in ring:
        #         ring.append(j)
    for ring in [ring]:
        cid = ring[0]
        subgraph = G.subgraph(ring)
        radius_sum = 0
        node_11 = subgraph.nodes[cid]
        x = node_11['x']
        y = node_11['y']
        for (id, data) in subgraph.nodes.data():
            if id != cid:
                subgraph.nodes[id]['r'] = math.sqrt((data['x'] - x)**2 + (data['y'] - y)**2)
                radius_sum += subgraph.nodes[id]['r']
        radius_mean = radius_sum / (len(ring) - 1)
        j = 0
        for (id, data) in subgraph.nodes.data():
            if id != cid:
                subgraph.nodes[id]['x'] = x + 10.0 * np.sin(2.0 * np.pi * j / (len(ring)-1))
                subgraph.nodes[id]['y'] = y + 10.0 * np.cos(2.0 * np.pi * j / (len(ring)-1))
                j += 1
            else:
                subgraph.nodes[id]['x'] = x
                subgraph.nodes[id]['y'] = y
            # G.nodes[id]['x'] = subgraph.nodes[id]['x']
            # G.nodes[id]['y'] = subgraph.nodes[id]['y']
        js_graph = json_graph.node_link_data(subgraph)
        with open(prefix + 'subgraph.json', 'w') as f:
            json.dump(js_graph, f)


if __name__ == '__main__':
    prefix = './data/bn-mouse-kasthuri/'
    # prefix = './data/VIS/'

    # main(prefix)
    G = load_json_graph(prefix + 'graph-with-pos.json')
    # G = load_json_graph(prefix + 'surr_graph.json')
    subgraph0 = load_json_graph(prefix + 'subgraph.json')

    nodes_data = list(subgraph0.nodes.data())
    for i in range(0, len(list(subgraph0.nodes))):
        subgraph0.nodes[nodes_data[i][0]]['x'] *= 10
        subgraph0.nodes[nodes_data[i][0]]['y'] *= 10

    # sub = G.subgraph([int(id) for id in subgraph0.nodes])
    # new_subgraph = {
    #     "nodes": [],
    #     "edges": []
    # }
    # nodes_data = list(sub.nodes.data())
    # for i in range(0, len(list(sub.nodes))):
    #     new_subgraph['nodes'].append({
    #         'id': nodes_data[i][0],
    #         'x': nodes_data[i][1]['x'],
    #         'y': nodes_data[i][1]['y']
    #     })
    # for i in range(0, len(list(sub.edges))):
    #     new_subgraph['edges'].append({
    #         'source': list(sub.edges)[i][0],
    #         'target': list(sub.edges)[i][1],
    #     })
    # print(new_subgraph)

    G = Graph(G)
    subgraph0 = Graph(subgraph0)
    # subgraph1 = Graph(load_json_graph(prefix + 'sub0.json'))

    # fuse and fuse_v2 are both available, the parameters are the same;
    # ! please try both fuse and fuse_v2 to test the effect
    # the first parameter is the entire graph
    # the second parameter is a list of the subgraphs to be fused in
    # the third parameter is a list of boolean value which defines whether the subgraph i needs for normalize
    # the forth parameter is a list of weights which determines each subgraph's weight, more weighted, more strong the constrains are.
    # G = fuse(G, [subgraph0], [True, True], [15, 15])
    # save_json_graph(G.to_networkx(), prefix + 'new.json')

    new_G, _new_G = fuse_v4(G, [subgraph0], [True, True], [150, 150])
    save_json_graph(new_G.to_networkx(), prefix + 'new.json')
    save_json_graph(_new_G.to_networkx(), prefix + '_new.json')