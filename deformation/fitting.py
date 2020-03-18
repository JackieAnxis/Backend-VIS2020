# -*- coding: UTF-8
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from deformation.Graph import Graph
from deformation.utils import load_json_graph, save_json_graph

def minimize_marker_distance(source_G, target_G, markers):
    adj_matrix = target_G.compute_adj_matrix()
    lap_matrix = target_G.laplacian_matrix(adj_matrix)

    
    # for i in range(target_G.nodes.shape[0]):
    #     for j in range(markers.shape[0]):

def aligning(source_G, target_G, markers):
    # R = [[ s, h, x]
    #      [-h, s, y]
    #      [ 0, 0, 1]]
    # X = [s, h, tx, ty]
    # [x  y 0 1]
    # [y -x 1 0] dot [s h x y].T
    #     ...
    k = 4 # [s, h, tx, ty]
    M = np.zeros((markers.shape[0] * 2, k))
    C = np.zeros((markers.shape[0] * 2, 1))

    source_marker_nodes = source_G.nodes[markers[:, 0], :]
    target_marker_nodes = target_G.nodes[markers[:, 1], :]
    for i in range(0, markers.shape[0]):
        x = target_marker_nodes[i][0]
        y = target_marker_nodes[i][1]
        M[i * 2] = np.array([x, y, 1, 0])
        M[i * 2 + 1] = np.array([y, -x, 0, 1])
        C[i * 2:i * 2 + 2] = source_marker_nodes[i][:, np.newaxis]

    X = sparse.linalg.lsqr(M, C, iter_lim=30000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)

    s = X[0][0]
    h = X[0][1]
    t = X[0][2:]
    R = np.array([[s, h], [-h, s]])
    return R, t

def fitting(source_G, target_G, markers):
    source_marker_nodes = source_G.nodes[markers[:, 0], :].T  # 2 * n
    target_marker_nodes = target_G.nodes[markers[:, 1], :].T  #v 2 * n
    source_center = np.mean(source_marker_nodes, axis=1)
    target_center = np.mean(target_marker_nodes, axis=1)

    h = 0
    s = 1
    t = (source_center - target_center)

    # R.dot(X)+t = X'
    x0 = np.zeros((4,))  # R.size / 2 + t.size / 2
    x0[0] = h
    x0[1] = s
    x0[2:4] = t

    def resSimXform(x, A, B):
        h = x[0]
        s = x[1]
        t = x[2:4]
        R = np.array([[s, h], [-h, s]])
        rot_A = R.dot(A) + t[:, np.newaxis]
        result = np.sqrt(np.sum((B - rot_A) ** 2, axis=0))
        return result

    b = least_squares(fun=resSimXform, x0=x0, jac='2-point', method='lm',
                      args=(target_marker_nodes, source_marker_nodes),
                      ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)

    h = b.x[0]
    s = b.x[1]
    t = b.x[2:4]
    R = np.array([[s, h], [-h, s]])

    return R, t


def length_minimize(source_G, target_G, marker):
    x0 = target_G.nodes.flatten()
    x0[marker[:, 1] * 2] = source_G.nodes[marker[:, 0]][:, 0]
    x0[marker[:, 1] * 2 + 1] = source_G.nodes[marker[:, 0]][:, 1]

    pair_index = []
    pair_expected_length = []
    for i in range(0, target_G.nodes.shape[0]):
        for j in range(i + 1, target_G.nodes.shape[0]):
            if target_G.rawgraph.has_edge(target_G.index2id[i], target_G.index2id[j]):
            # if i in marker[:, 1] and j in marker[:, 1]:
            #     source_marker_id_i = marker[:, 0][list(marker[:, 1]).index(i)]
            #     source_marker_id_j = marker[:, 0][list(marker[:, 1]).index(j)]
            #     pair_expected_length.append(source_G.nodes[source_marker_id_i] - source_G.nodes[source_marker_id_j])
            #     pair_index.append([i, j, 1])
            # else:
                pair_expected_length.append(target_G.nodes[i] - target_G.nodes[j])
                pair_index.append([i, j, 1])

    pair_index = np.array(pair_index, dtype=np.int32)
    pair_expected_length = np.sqrt(np.sum(np.array(pair_expected_length) ** 2, axis=1)) * pair_index[:, 2]

    def resSimXform(x, A, B):
        x = x.reshape(((int(x.shape[0] / 2), 2)))
        edge_lengths = np.sqrt(np.sum((x[A[:, 0]] - x[A[:, 1]]) ** 2, axis=1))
        E1 = np.sum((edge_lengths / edge_lengths[0] - B / B[0]) ** 2)
        E2 = np.sum(np.sqrt(np.sum((x[marker[:, 1]] - source_G.nodes[marker[:, 0]])**2, axis=1)))
        return E1 + E2

    b = least_squares(fun=resSimXform, x0=x0, jac='2-point', method='trf', args=(pair_index, pair_expected_length),
                      ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)

    return b.x


def El_linear_system(source_G, target_G, marker, wl = 1):
    target_G.compute_adjacent_matrix()
    L = target_G.rw_laplacian_matrix(target_G.adj_matrix)
    V = target_G.nodes
    Delta = L.dot(V)
    # n = L.shape[0]
    n = target_G.nodes.shape[0]

    # LS = np.zeros([3*n, 3*n])
    # LS[0*n:1*n, 0*n:1*n] = (-1) * L
    # LS[1*n:2*n, 1*n:2*n] = (-1) * L
    # LS[2*n:3*n, 2*n:3*n] = (-1) * L
    C = np.zeros(((n + marker.shape[0]) * 2, 1))
    index = np.arange(L.shape[0], dtype=np.int32)
    LS = np.zeros([2 * n, 2 * n])
    for i in index:
        LS[i * 2, index * 2] = (-1) * L[i]
        LS[i * 2 + 1, index * 2 + 1] = (-1) * L[i]

    # for i in range(n):
    for i in range(L.shape[0]):
        nb_idx = target_G.get_local_neighbor(i, target_G.adj_matrix)  # node i and its neibors
        ring = np.array([i] + nb_idx)  # node i and its neibors in subgraph
        V_ring = V[ring]
        n_ring = V_ring.shape[0]

        A = np.zeros([n_ring * 2, 4])
        for j in range(n_ring):
            A[j] = [V_ring[j, 0], -V_ring[j, 1], 1, 0]
            A[j + n_ring] = [V_ring[j, 1], V_ring[j, 0], 0, 1]

        # Moore-Penrose Inversion
        A_pinv_s = np.linalg.pinv(A)
        A_pinv = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A))
        s = A_pinv[0]
        h = A_pinv[1]
        # t = A_pinv[2:4]

        T_delta = np.vstack([
            Delta[i, 0] * s - Delta[i, 1] * h,
            Delta[i, 0] * h + Delta[i, 1] * s
        ])

        LS[i * 2, np.hstack([ring * 2, ring * 2 + 1])] += T_delta[0]
        LS[i * 2 + 1, np.hstack([ring * 2, ring * 2 + 1])] += T_delta[1]
        # LS[i+2*n, np.hstack([ring, ring+n, ring+2*n])] += T_delta[2]

    constraint_coef = []

    # Handle constraints
    for i in range(0, marker.shape[0]):
        source_marker_id = marker[i, 0]
        target_marker_id = marker[i, 1]
        constraint_coef.append(np.arange(2 * n) == target_marker_id * 2)
        constraint_coef.append(np.arange(2 * n) == target_marker_id * 2 + 1)
        C[n * 2 + i * 2:n * 2 + i * 2 + 2, :] = source_G.nodes[source_marker_id][:, np.newaxis]

    constraint_coef = np.matrix(constraint_coef)

    A = np.vstack([wl * LS, 2 * wl * constraint_coef])
    M = sparse.coo_matrix(A)
    C = 2 * wl * C
    return M, C


if __name__ == '__main__':
    prefix = './data/power-662-bus/'

    source = load_json_graph(prefix + 'source.json')
    raw_target = load_json_graph(prefix + 'target0.json')

    # markers = np.array([[575, 257], [574, 222], [476, 245], [588, 181]])
    markers = np.array([[575, 222], [466, 195], [477, 257]])

    target = raw_target.copy()
    # target_nodes = list(target.nodes.data())
    # for i in range(len(target_nodes)):
    #     for j in range(i + 1, len(target_nodes)):
    #         if not target.has_edge(target_nodes[i][0], target_nodes[j][0]):
    #             target.add_edge(target_nodes[i][0], target_nodes[j][0])


    source_G = Graph(source)
    source_G.normalize()
    target_G = Graph(target)
    markers[:, 0] = np.array([source_G.id2index[str(marker)] for marker in markers[:, 0]])
    markers[:, 1] = np.array([target_G.id2index[str(marker)] for marker in markers[:, 1]])
    print('fitting...')
    # R, t = fitting(source_G, target_G, markers)
    R, t = aligning(source_G, target_G, markers)
    target_G.nodes = target_G.nodes.dot(R.T) + t
    print('Minimize...')
    # X = length_minimize(source_G, target_G, markers)
    # target_G.nodes = X.reshape((target_G.nodes.shape[0], 2))
    # smooth
    M, C = El_linear_system(source_G, target_G, markers)
    X = sparse.linalg.lsqr(M, C, iter_lim=30000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
    # target_G.nodes = X[0].reshape((target_G.nodes.shape[0], 2))[0:target_G.nodes.shape[0], :]
    print('!')
    for node in raw_target.nodes:
        index = target_G.id2index[str(node)]
        raw_target.nodes[node]['x'] = target_G.nodes[index][0]
        raw_target.nodes[node]['y'] = target_G.nodes[index][1]

    save_json_graph(raw_target, prefix + '_target0.json')