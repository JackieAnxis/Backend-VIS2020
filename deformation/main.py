# -*- coding: UTF-8
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scipy import sparse
import numpy as np
import networkx as nx
from scipy.optimize import least_squares
from deformation.fuse import fuse_main
from deformation.utils import load_json_graph, save_json_graph
from deformation.correspondence import non_rigid_registration, build_correspondence
from deformation.Graph import Graph
from deformation.deform import aligning

def affine_minimize(source_G, target_G, deformed_source_G, correspondences):
    target_G.build_elementary_cell()
    Q = []
    for i in range(0, source_G.edges.shape[0]):
        Q.append(deformed_source_G.V[i].dot(np.linalg.inv(source_G.V[i]))) # edge num * (2 x 2) ?
    A = target_G.A # TODO: A?
    correspondence_count = 0
    for correspondence in correspondences:
        if correspondence.shape[0] > 0:
            correspondence_count += correspondence.shape[0]
        else:
            correspondence_count += 1

    C = np.zeros((correspondence_count * 4, 1)) # [Q_i_11, Q_i_12, Q_i_21, Q_i_22].T
    I = np.zeros((correspondence_count * 4 * 3, 3)) # TODO: I 的行列定位需要重新看一下
    offset = 0
    for i in range(0, len(correspondences)):
        correspondence = correspondences[i]
        target_edge = target_G.new_edges[i, :]
        if correspondence.shape[0] > 0:
            for j in range(0, correspondence.shape[0]):
                for k in range(0, 2): # for x to y
                    row_index = np.tile(np.linspace(0, 1, 2, dtype=np.int32) + offset + j * 2 * 2 + k * 2, [3, 1]).T
                    col_index = np.tile(target_edge * 2 + k, [2, 1]) # 2 represents x and y
                    value = A[i]
                    row_index_of_I = np.array(range(0, 6)) + offset * 3 + j * 2 * 2 * 3 + k * 2 * 3
                    I[row_index_of_I, :] = \
                        np.hstack((row_index.flatten()[:, np.newaxis],
                                   col_index.flatten()[:, np.newaxis],
                                   value.flatten()[:, np.newaxis]))
                row_index_of_C = np.array(range(0, 4)) + offset
                C[row_index_of_C, 0] = Q[int(correspondence[j])].flatten() # [1, 2, 3, 4] 将所有的已知的source排成一列
            offset += 2 * 2 * correspondence.shape[0]
        else:
            for k in range(0, 2):  # for x to y
                row_index = np.tile(np.linspace(0, 1, 2, dtype=np.int32) + offset + k * 2, [3, 1]).T
                col_index = np.tile(target_edge * 2 + k, [2, 1])  # 2 represents x and y
                value = A[i]
                row_index_of_I = np.array(range(0, 6)) + offset * 3 + k * 2 * 3
                I[row_index_of_I, :] = \
                    np.hstack((row_index.flatten()[:, np.newaxis],
                               col_index.flatten()[:, np.newaxis],
                               value.flatten()[:, np.newaxis]))
            row_index_of_C = np.array(range(0, 4)) + offset
            # C[row_index_of_C, 0] = np.zeros(4).flatten() # [1, 0, 0, 1]
            C[row_index_of_C, 0] = np.eye(2).flatten() # [1, 0, 0, 1]
            offset += 2 * 2
    # M * x = C
    # x = deformded target nodes position
    # x = [..., x_j, y_j, z_j, ...].T
    M = sparse.coo_matrix((I[:, 2], (I[:, 0], I[:, 1])), shape=(2 * 2 * correspondence_count, 2 * target_G.new_nodes.shape[0]))
    return M, C
    
def position_minimize(source_G, target_G, deformed_source_G, correspondences):
    M = np.zeros((target_G.new_nodes.shape[0] * 2, target_G.new_nodes.shape[0] * 2))
    C = np.zeros((target_G.new_nodes.shape[0] * 2, 1))
    # C = np.zeros((deformded_source_G.new_nodes.shape[0] * 2, 1))
    for i in range(0, len(correspondences)):
        corr = correspondences[i]
        for j in range(0, corr.shape[0]):
            deformed_source_node = deformed_source_G.new_nodes[corr[j]]
            M[i * 2, i * 2] += 1
            M[i * 2 + 1, i * 2 + 1] += 1
            C[i * 2 : i * 2 + 2] += deformed_source_node[:, np.newaxis]
    M = sparse.coo_matrix(M)
    return M, C

def direction_minimize(source_G, target_G, deformed_source_G, node_corrs, edge_corrs):
    edge_corrs_count = 0
    for edge_corr in edge_corrs:
        if edge_corr.shape[0] > 0:
            edge_corrs_count += edge_corr.shape[0]
        else:
            edge_corrs_count += 1

    n = target_G.new_nodes.shape[0]
    C = np.zeros((n * (n-1), 1))
    M = np.zeros((n * (n-1), n * 2))
    offset = 0
    for i in range(0, target_G.nodes.shape[0]):
        target_node_0 = target_G.nodes[i]
        source_node_0 = np.mean(deformed_source_G.nodes[node_corrs[i]], axis=0)
        for j in range(i + 1, target_G.nodes.shape[0]):
            target_node_1 = target_G.nodes[j]
            source_node_1 = np.mean(deformed_source_G.nodes[node_corrs[j]], axis=0)
            M[offset, i * 2] = target_node_0[0]
            M[offset, j * 2] = -target_node_1[0]
            M[offset + 1, i * 2 + 1] = target_node_0[1]
            M[offset + 1, j * 2 + 1] = -target_node_1[1]
            C[offset: offset + 2] = (source_node_0 - source_node_1)[:, np.newaxis]
            offset += 2
    M = sparse.coo_matrix(M)
    return M, C

def distance_minimize(source_G, target_G, deformded_source_G, node_corrs, edge_corrs):
    def resSimXform(x, A, B):
        x = x.reshape(((int(x.shape[0] / 2), 2)))
        edges_length = (np.sqrt(np.sum((x[A[:, 0]] - x[A[:, 1]])**2, axis=1)))
        E = edges_length - B
        return E

    x0 = target_G.nodes.flatten()
    pair_index = []
    for i in range(0, target_G.nodes.shape[0]):
        for j in range(i + 1, target_G.nodes.shape[0]):
            pair_index.append([i, j])
    pair_index = np.array(pair_index, dtype=np.int32)
    pair_expected_length = np.zeros((pair_index.shape[0]))

    for k in range(0, pair_index.shape[0]):
        i = pair_index[k][0]
        j = pair_index[k][1]
        c_i = node_corrs[i] # an array
        c_j = node_corrs[j] # an array
        mean_direction = np.mean(deformded_source_G.nodes[c_i], axis=0) - np.mean(deformded_source_G.nodes[c_j], axis=0)
        mean_distance = np.sqrt(np.sum(mean_direction**2))
        pair_expected_length[k] = mean_distance

    b = least_squares(fun=resSimXform, x0=x0, jac='2-point', method='trf', args=(pair_index, pair_expected_length),
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)
    x = b.x
    
    y = x.reshape(((int(x.shape[0] / 2), 2)))
    print(y)
    print(np.sqrt(np.sum((y[pair_index[:, 0]] - y[pair_index[:, 1]])**2, axis=1)))
    
    return x

def deformation_transfer(source_G, target_G, deformed_source_G, correspondences):
    source_G.compute_adjacent_matrix()
    source_G.compute_third_node()
    deformed_source_G.compute_third_node()
    target_G.compute_third_node()

    corrs = []
    for i in range(0, target_G.edges.shape[0]):
        target_edge = target_G.edges[i]
        corr = []
        target_node_0 = int(target_edge[0])
        target_node_1 = int(target_edge[1])
        correspondence_0 = correspondences[target_node_0]
        correspondence_1 = correspondences[target_node_1]
        for j in range(0, correspondence_0.shape[0]):
            for k in range(0, correspondence_1.shape[0]):
                if source_G.adj_matrix[correspondence_0[j], correspondence_1[k]]:
                    source_edge_0 = [correspondence_0[j], correspondence_1[k]]
                    source_edge_1 = [correspondence_1[k], correspondence_0[j]]
                    source_edge_index = np.where(np.all(source_G.edges == source_edge_0, axis=1) + np.all(source_G.edges == source_edge_1, axis=1))[0][0]
                    corr.append(source_edge_index) # 找出在source graph中所有可能的对应的边（的index）
        corrs.append(np.array(corr, dtype=np.int32))
    
    edge_corrs = corrs
    node_corrs = correspondences

    # Ma, Ca = affine_minimize(source_G, target_G, deformed_source_G, edge_corrs)
    # Mp, Cp = position_minimize(source_G, target_G, deformed_source_G, node_corrs)
    # Md, Cd = direction_minimize(source_G, target_G, deformed_source_G, node_corrs, edge_corrs)
    # M = sparse.vstack([Ma*10, Mp*0, Md*0])
    # C = np.vstack([Ca*10, Cp*0, Cd*0])
    # x = sparse.linalg.lsqr(M, C, iter_lim=5000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)[0]
    
    x = distance_minimize(source_G, target_G, deformed_source_G, node_corrs, edge_corrs)
    deformed_target_G = target_G.copy()
    deformed_target_G.nodes = np.reshape(x, (int(x.shape[0] / 2), 2))[0:deformed_target_G.nodes.shape[0], :]
    return deformed_target_G

def generate(markers, source_graph, deformed_source_graph, target_graph, correspondence):
    # load source graph, deformed source graph and target graph
    prefix = './data/test_running/'
    
    source_G = Graph(source_graph)
    deformed_source_G = Graph(deformed_source_graph)
    target_G = Graph(target_graph)

    # if len(correspondence) >= 3:
    if len(correspondence) < 0:
        # deformation transfer 这段暂时先不用
        print('Use search correspondence...')
        correspondence = np.array(correspondence)
        deformed_target_G = deformation_transfer(source_G, target_G, deformed_source_G, correspondence)
    else:
        # build node to node correspondence by a few markers
        markers = np.array(markers)  # [source, target]
        K = 2
        max_dis = 0.3
        fine = False
        ws = 5.0  # smooth
        wi = 1.0
        wc = [1, 500, 3000, 5000]

        markers[:, 0] = np.array([source_G.id2index[str(id)] for id in markers[:, 0]])
        markers[:, 1] = np.array([target_G.id2index[str(id)] for id in markers[:, 1]])
        marker_increasing = True
        correspondence = np.zeros((0,2), dtype=np.int32)
        while marker_increasing:
            reg_source_G, reg_target_G, R, t = non_rigid_registration(source_G, target_G, ws, wi, wc, markers, K,
                                                                      max_dis)
            correspondence = build_correspondence(reg_source_G, reg_target_G, correspondence)
            if correspondence.shape[0] <= markers.shape[0]:
                marker_increasing = False
            markers = correspondence
            # return reg_target_G.to_networkx()

        markers = markers.copy()
        markers[:, 0] = np.array([source_G.index2id[index] for index in markers[:, 0]])
        markers[:, 1] = np.array([target_G.index2id[index] for index in markers[:, 1]])
        markers.tolist()
        for marker in markers:
            print('[', marker[0], ',', marker[1], '],')
        print('################')

        # return reg_target_G.to_networkx()

        markers = correspondence
        marker_increasing = True
        while marker_increasing:
            reg_source_G, reg_target_G, R, t = non_rigid_registration(deformed_source_G, target_G, ws, wi, wc,
                                                                      correspondence, K, max_dis)
            correspondence = build_correspondence(reg_source_G, reg_target_G, correspondence)
            if correspondence.shape[0] <= markers.shape[0]:
                marker_increasing = False
            markers = correspondence

        markers = markers.copy()
        markers[:, 0] = np.array([source_G.index2id[index] for index in markers[:, 0]])
        markers[:, 1] = np.array([target_G.index2id[index] for index in markers[:, 1]])
        markers.tolist()
        for marker in markers:
            print('[', marker[0], ',', marker[1], '],')
        print('################')


        # correspondence = [[] for i in range(target_G.nodes.shape[0])]
        # for marker in markers:
        #     correspondence[marker[1]].append(marker[0])
        # correspondence = [np.array(corr) for corr in correspondence]
        # reg_target_G = deformation_transfer(source_G, target_G, deformed_source_G, correspondence)

        # R, t = aligning(target_G, reg_target_G,
        #                           np.array([[index, index] for index in range(target_G.nodes.shape[0])]))
        # reg_target_G.nodes = reg_target_G.nodes.dot(R.T) + t

        return reg_target_G.to_networkx()

def main_for_power():
    # load source graph, deformed source graph and target graph
    # prefix = './data/road-euroroad/'
    prefix = './data/power-662-bus/'
    G = load_json_graph(prefix + 'graph-with-pos.json')
    cycles = nx.cycle_basis(G)
    cycles = sorted(cycles, key = lambda c: len(c), reverse=True)
    # cycles = list(filter(lambda c: len(c) > 0 and len(c) < 20, cycles))

    #### road-euroroad
    # source_nodes = [86, 87, 88, 91, 135, 250, 249, 196, 193, 195, 192, 132, 126, 124]
    # target_nodes = [353, 352, 351, 350, 347, 346, 349, 787, 786, 785, 575, 572, 573, 576, 577, 579, 357, 355]
    # markers = [[192, 347], [86, 785], [91, 576], [249, 355]]
    ####

    #### power-662-bus
    source_nodes = [462, 575, 589, 588, 477, 476, 466, 574]
    target_nodes = [
        [222, 220, 221, 257, 195, 194, 181, 182, 183, 245, 246],
        [482, 487, 580, 583, 488],
        [135, 136, 11, 10, 12, 271, 273, 289, 290, 137],
        [28, 30, 228, 306, 60, 59, 61, 317, 31],
        [272, 271, 216, 8, 7, 9, 71, 99, 214, 215, 320],
    ]
    source_top_node = source_nodes[0]
    target_top_node = [nodes[0] for nodes in target_nodes]
    ####

    source = nx.Graph(G.subgraph(source_nodes))
    save_json_graph(source, prefix + 'source.json')
    # save_json_graph(target, prefix + 'target.json')

    markers = []
    source_top_node_index = source_nodes.index(source_top_node)
    for k in range(0, len(target_nodes)):
        target_top_node_index = target_nodes[k].index(target_top_node[k])
        marker = []
        for i in range(0, 4):
            source_marker_index = int(source_top_node_index + i * len(source_nodes) / 4) % len(source_nodes)
            target_marker_index = int(target_top_node_index + i * len(target_nodes[k]) / 4) % len(target_nodes[k])
            source_marker = source_nodes[source_marker_index]
            target_marker = target_nodes[k][target_marker_index]
            marker.append([source_marker, target_marker])
        markers.append(marker)

    #####
    # [548, 47], [536, 45], [532, 76], [528, 82], [531, 155], [584, 227]

    markers = [
        # [[462, 246], [589, 220], [466, 182], [477, 194]],
        # [[462, 488], [589, 482], [466, 583], [477, 487]],
        # [[462, 137], [589, 273], [466, 11], [477, 12]],
        # [[462, 317], [589, 59], [466, 28], [477, 228]],
        # [[462, 71], [589, 7], [466, 215], [477, 272]],
        [[462, 220], [589, 257], [466, 245], [477, 181]],
        [[462, 583], [589, 488], [466, 580], [477, 482]],
        [[462, 290], [589, 271], [466, 135], [477, 10]],
        [[462, 317], [589, 60], [466, 28], [477, 228]],
        # [[462, 8], [589, 271], [466, 71], [477, 215]],
        [[462, 99], [589, 7], [466, 215], [477, 272]],
    ]
    #####

    G = Graph(G)
    # deform the source graph #
    cycle_source_nodes = np.array([G.id2index[str(id)] for id in source_nodes])
    center = np.mean(G.nodes[cycle_source_nodes], axis=0)
    radius = np.mean(np.sqrt(np.sum((G.nodes[cycle_source_nodes] - center)**2, axis=1)))

    deformed_source = source.copy()
    begin_node = G.id2index[str(source_top_node)]
    begin_index = np.nonzero(cycle_source_nodes == begin_node)[0]
    interval = 2.0 * np.pi / (cycle_source_nodes.shape[0])
    for i in range(0, cycle_source_nodes.shape[0]):
        index = int((begin_index + i) % cycle_source_nodes.shape[0])
        x =  center[0] + radius * np.sin(interval * i)
        y =  center[1] - radius * np.cos(interval * i)
        id = int(G.index2id[cycle_source_nodes[index]])
        deformed_source.nodes[id]['x'] = x
        deformed_source.nodes[id]['y'] = y

    save_json_graph(deformed_source, prefix + '_source.json')

    deformed_targets = [deformed_source]
    for k in range(0, len(target_nodes)):
        target = nx.Graph(nx.subgraph(G.rawgraph, [str(id) for id in target_nodes[k]]))
        save_json_graph(target, prefix + 'target' + str(k) + '.json')
        correspondence = []
        deformed_target = generate(markers[k], source, deformed_source, target, correspondence)
        deformed_targets.append(deformed_target)
        save_json_graph(deformed_target, prefix + '_target' + str(k) + '.json')
        # save_json_graph(nx.union(deformed_target, deformed_source), prefix + '_target' + str(k) + '.json')
    fused_graph = fuse_main(G.rawgraph, deformed_targets)
    save_json_graph(fused_graph, prefix + 'new.json')


def main_for_mouse():
    # load source graph, deformed source graph and target graph
    prefix = './data/bn-mouse-kasthuri/'
    G = load_json_graph(prefix + 'graph-with-pos.json')

    #### power-662-bus
    source_nodes = [720, 941, 943, 939, 942, 944, 940]
    target_nodes = [
        [676, 853, 850, 851, 854, 848, 852, 849, 855],
        [703, 920, 921, 919],
        [660, 782, 780, 784, 789, 786, 781, 787, 783]
    ]
    source_top_node = source_nodes[0]
    target_top_node = [nodes[0] for nodes in target_nodes]
    ####

    source = nx.Graph(G.subgraph(source_nodes))
    save_json_graph(source, prefix + 'source.json')

    markers = [[[source_nodes[i], target_nodes[k][i]] for i in [0, 1, -1]] for k in range(len(target_nodes))]
    #####

    G = Graph(G)
    # deform the source graph #
    cycle_source_nodes = np.array([G.id2index[str(id)] for id in source_nodes[1:]])
    center = G.nodes[G.id2index[str(source_nodes[0])]] # np.mean(G.nodes[cycle_source_nodes], axis=0)
    radius = np.mean(np.sqrt(np.sum((G.nodes[cycle_source_nodes] - center) ** 2, axis=1)))

    deformed_source = source.copy()
    begin_node = cycle_source_nodes[0]
    begin_index = np.nonzero(cycle_source_nodes == begin_node)[0]
    interval = 1.0 * np.pi / (cycle_source_nodes.shape[0])
    for i in range(0, cycle_source_nodes.shape[0]):
        index = int((begin_index + i) % cycle_source_nodes.shape[0])
        x = center[0] + radius * np.sin(interval )# * i)
        y = center[1] - radius * np.cos(interval )# * i)
        id = int(G.index2id[cycle_source_nodes[index]])
        deformed_source.nodes[id]['x'] = x
        deformed_source.nodes[id]['y'] = y

    save_json_graph(deformed_source, prefix + '_source.json')

    deformed_targets = [deformed_source]
    for k in range(0, len(target_nodes)):
        target = nx.Graph(nx.subgraph(G.rawgraph, [str(id) for id in target_nodes[k]]))
        save_json_graph(target, prefix + 'target' + str(k) + '.json')
        correspondence = []
        deformed_target = generate(markers[k], source, deformed_source, target, correspondence)
        deformed_targets.append(deformed_target)
        save_json_graph(deformed_target, prefix + '_target' + str(k) + '.json')
    fused_graph = fuse_main(G.rawgraph, deformed_targets)
    save_json_graph(fused_graph, prefix + 'new.json')


if __name__ == '__main__':
    main_for_power()
    # main_for_mouse()
