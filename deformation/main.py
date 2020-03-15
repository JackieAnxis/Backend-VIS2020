# -*- coding: UTF-8
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scipy import sparse
import numpy as np
from scipy.optimize import least_squares
# from utils import save_pickle_file, load_pickle_file, load_json_graph, save_json_graph
# from rotate import calculate_T, rotate, third_node
# from correspondence import non_rigid_registration, build_correspondence
# from Graph import Graph
from deformation.utils import save_pickle_file, load_pickle_file, load_json_graph, save_json_graph
from deformation.rotate import calculate_T, rotate, third_node
from deformation.correspondence import non_rigid_registration, build_correspondence, Ec_linear_system
from deformation.Graph import Graph

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
        return (np.sqrt(np.sum((x[A[:, 0]] - x[A[:, 1]])**2, axis=1))) - B
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

def generate(markers, source_graph, deformed_source_graph, target_graph, correspondence, save_to_running = True):
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
        marker = np.array(markers)  # [source, target]
        # marker = np.array([[0, 0], [2, 2], [4, 4]])  # [source, target]
        K = 2
        max_dis = 0.3
        fine = False
        ws = 5.0  # smooth
        wi = 1.0
        wc = [1, 500, 3000, 5000]
        reg_source_G, reg_target_G, R, t = non_rigid_registration(source_G, target_G, ws, wi, wc, marker, K, max_dis)
        iterations_count = 0
        max_iterations_count = 10
        while not fine:
            correspondence = build_correspondence(reg_source_G, reg_target_G, K, max_dis)
            iterations_count += 1
            if iterations_count > max_iterations_count:
                print('Too musch iterations on building correspondence')
            for cor in correspondence:
                if cor.shape[0] < 1:  # some target node has no correspondence
                    if K > 5:
                        max_dis += 0.1
                        fine = False
                        break
                    K += 1
                    fine = False
                    break
                else:
                    fine = True
        print('correspondence:')
        print(correspondence)
        # deformation transfer
        deformed_target_G = deformation_transfer(source_G, target_G, deformed_source_G, correspondence)
        
        # simply calcualte the mean position
        # deformed_target_G = target_G.copy()
        # for i in range(0, len(correspondence)):
        #     deformed_target_G.nodes[i, :] = np.mean(deformed_source_G.nodes[correspondence[i], :], axis=0)

    if save_to_running:
        save_json_graph(source_G.to_networkx(), prefix + 'source.json')
        save_json_graph(target_G.to_networkx(), prefix + 'target.json')
        save_json_graph(deformed_source_G.to_networkx(), prefix + '_source.json')
        save_json_graph(deformed_target_G.to_networkx(), prefix + '_target.json')
    return deformed_target_G.to_networkx()

def main():
    # load source graph, deformed source graph and target graph
    prefix = './data/test_running/'
    source_graph = load_json_graph(prefix + 'source.json')
    deformed_source_graph = load_json_graph(prefix + '_source.json')
    target_graph = load_json_graph(prefix + 'target.json')

    markers = [[64, 77], [34, 84], [104, 112]]
    correspondence = []
    deformed_target_graph_networkx = generate(markers, source_graph, deformed_source_graph, target_graph, correspondence, save_to_running=False)
    save_json_graph(deformed_target_graph_networkx, prefix + '_target.json')



if __name__ == '__main__':
    main()
