# -*- coding: UTF-8
import json
import csv
import os
import random
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from MT.deform import non_rigid_registration, aligning
from MT.correspondence import compute_distance_matrix, build_correspondence_v1, build_correspondence_v2, build_correspondence_v3, build_correspondence_v4
from MT.Graph import Graph
from MT.optimization import merge
from models.utils import load_json_graph, save_json_graph
from fgm.fgm import fgm
from models.layout import tree_layout, radial_tree_layout, layout, MMM_layout, GEM_layout, nx_spring_layout, SM_layout, remove_overlap

# names = ["Ga", "Pm", "Sm", "Smac", "Rrwm", "FgmU"] # ,"IpfpU", "IpfpS", "FgmD"]
names = ["FgmU"] # ,"IpfpU", "IpfpS", "FgmD"]

def markers2matrix(markers, n, m):
    mat = np.zeros((n, m))
    for marker in markers:
        mat[marker[0], marker[1]] = 1
    return mat

def mean_link_length_of_nodes(G):
    A = G.compute_adjacent_matrix()
    MLL = [] # mean link length
    for i in range(A.shape[0]):
        d = 0
        e = 0
        for j in range(0, A.shape[0]):
            if A[i, j]:
                d += np.linalg.norm(G.nodes[i] - G.nodes[j])
                e += 1
        MLL.append(d / e)
    return np.array(MLL)

def mean_link_length(G):
    A = G.compute_adjacent_matrix()
    d = 0
    e = 0
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            if A[i, j]:
                d += np.linalg.norm(G.nodes[i] - G.nodes[j])
                e += 1
    d /= e
    return d

def scale(G0, G1):
    V = G1.nodes.copy()
    raw_max = np.max(G0.nodes, axis=0)
    raw_min = np.min(G0.nodes, axis=0)
    max = np.max(V, axis=0)
    min = np.min(V, axis=0)
    scale = np.mean((raw_max - raw_min) / (max - min)) * 0.8
    V -= min
    V *= scale
    V += raw_min
    V -= (np.max(V, axis=0) + np.min(V, axis=0)) / 2
    V += (np.max(G0.nodes, axis=0) + np.min(G0.nodes, axis=0)) / 2
    return V

def interpolate_v2(source_G, deformed_source_G, sequence):
    results = [source_G]
    for id in sequence:
        i = source_G.id2index[str(id)]
        last = results[len(results) - 1]
        inter = last.copy()
        inter.nodes = last.nodes.copy()
        inter.nodes[i] = deformed_source_G.nodes[i]
        results.append(inter)
    return results

def interpolate_v1(source_G, deformed_source_G, n):
    results = []
    for i in range(n):
        inter = source_G.copy()
        inter.nodes = source_G.nodes * 1.0 * (n - 1 - i) / (n-1) + deformed_source_G.nodes * 1.0 * i / (n-1)
        results.append(inter)
    return results

def generate(source_graph, deformed_source_graph, target_graph, markers=[]):
    ### convert class ###
    source_G = Graph(source_graph)
    deformed_source_G = Graph(deformed_source_graph)
    target_G = Graph(target_graph)

    ### convert id markers into index markers ###
    if len(markers) > 0:
        markers = np.array(markers)
        markers[:, 0] = np.array([source_G.id2index[str(id)] for id in markers[:, 0]])
        markers[:, 1] = np.array([target_G.id2index[str(id)] for id in markers[:, 1]])

    deformed_target_Gs, markers = generate_G(source_G, deformed_source_G, target_G, markers)

    markers[:, 0] = np.array([source_G.index2id[index] for index in markers[:, 0]])
    markers[:, 1] = np.array([target_G.index2id[index] for index in markers[:, 1]])

    deformed_target_G = deformed_target_Gs['Rrwm']['deformed_target_G']

    return deformed_target_G.to_networkx(), markers.tolist()

def generate_G(source_G, deformed_source_G, target_G, given_markers=[]):
    ### store raw data ###
    raw_source_G = source_G.copy()
    raw_deformed_source_G = deformed_source_G.copy()
    raw_target_G = target_G.copy()

    ### force directed layout ###
    fm3_source = nx_spring_layout(source_G.to_networkx())
    fm3_target = nx_spring_layout(target_G.to_networkx())
    origin_fm3_source_G = Graph(fm3_source)
    origin_fm3_target_G = Graph(fm3_target)

    if len(given_markers) == 0:
        G1_node_link_data = json.dumps(json_graph.node_link_data(fm3_source))
        G2_node_link_data = json.dumps(json_graph.node_link_data(fm3_target))
        M = fgm(G1_node_link_data, G2_node_link_data)

    deformed_target_Gs = {}
    names = ['Rrwm']
    for name in names:
        fm3_source_G = origin_fm3_source_G.copy()
        fm3_target_G = origin_fm3_target_G.copy()
        if len(given_markers) == 0:
            generated_markers = M[name]
            generated_markers[:, 0] = np.array([source_G.id2index[str(id)] for id in generated_markers[:, 0]])
            generated_markers[:, 1] = np.array([target_G.id2index[str(id)] for id in generated_markers[:, 1]])
            markers = generated_markers
        else:
            markers = given_markers
        ### align and flip to fit two force directed layout###
        R, t = aligning(fm3_source_G, fm3_target_G, markers)
        fm3_target_G.nodes = fm3_target_G.nodes.dot(R.T) + t
        distance_matrix = compute_distance_matrix(fm3_target_G.nodes, fm3_source_G.nodes)
        min_dis_sum = np.sum(np.min(distance_matrix, axis=1))
        flip_axis = -1
        # i: flip axis, 0 represent flip by x=0, 1 represent flip by y=0
        for i in range(2):
            fm3_target_G_copy = fm3_target_G.copy()
            # flip
            fm3_target_G_copy.nodes[:, i] = np.mean(fm3_target_G_copy.nodes[:, i]) - (fm3_target_G_copy.nodes[:, i] - np.mean(fm3_target_G_copy.nodes[:, i]))
            # align again
            R, t = aligning(fm3_source_G, fm3_target_G_copy, markers)
            fm3_target_G_copy.nodes = fm3_target_G_copy.nodes.dot(R.T) + t
            distance_matrix_copy = compute_distance_matrix(fm3_target_G_copy.nodes, fm3_source_G.nodes)
            min_dis_sum_flip = np.sum(np.min(distance_matrix_copy, axis=1))
            if min_dis_sum_flip < min_dis_sum:
                flip_axis = i
                min_dis_sum = min_dis_sum_flip
                fm3_target_G = fm3_target_G_copy
                distance_matrix = distance_matrix_copy

        if len(given_markers) == 0:
            markers_matrix = markers2matrix(markers, source_G.nodes.shape[0], target_G.nodes.shape[0])
            SA = source_G.compute_adjacent_matrix()
            TA = target_G.compute_adjacent_matrix()
            new_markers = []
            neighbor_rate_threshold = 0.7
            distance_rate_threshold = 1.5
            new_marker_rate_range = [0.1, 0.5]
            eps = 1e-6
            while True:
                new_markers = []
                for marker_pair in markers:
                    sm = marker_pair[0] # source marker
                    tm = marker_pair[1] # target marker
                    smn = np.nonzero(SA[sm])[0] # source marker neighbors
                    tmn = np.nonzero(TA[tm])[0] # target marker neighbors
                    smn2tm = np.nonzero(np.sum(markers_matrix[smn, :], axis=0))[0] # source marker neighbors' correspond target markers
                    smn2tms = set(smn2tm)  # source marker neighbors' correspond target markers set
                    tmns = set(tmn) # target marker neighbors set
                    and_count = len(smn2tms & tmns)
                    smmll = np.mean(np.sqrt(np.sum((source_G.nodes[[sm for i in range(len(smn))]] - source_G.nodes[smn]) ** 2, axis=1))) # source markers' mean edge length
                    tmmll = np.mean(np.sqrt(np.sum((target_G.nodes[[tm for i in range(len(tmn))]] - target_G.nodes[tmn]) ** 2, axis=1)))  # target markers' mean edge length
                    ### delete markers that links too much different neighbors ###
                    if and_count < len(smn2tms) * neighbor_rate_threshold and and_count < len(tmn) * neighbor_rate_threshold:
                        # need to be deleted
                        continue
                    ### delete markers that are far ###
                    dis = np.sqrt(np.sum((fm3_source_G.nodes[sm] - fm3_target_G.nodes[tm])**2))
                    if dis > smmll * distance_rate_threshold or dis > tmmll * distance_rate_threshold:
                        continue
                    new_markers.append(marker_pair.tolist())

                new_marker_rate = len(new_markers) / len(markers)
                if new_marker_rate < new_marker_rate_range[0]:
                    neighbor_rate_threshold -= 0.1
                    distance_rate_threshold += 0.1
                    neighbor_rate_threshold = np.max((0.3, neighbor_rate_threshold))
                    distance_rate_threshold = np.min((5, distance_rate_threshold))
                    if distance_rate_threshold >= 5 - eps and neighbor_rate_threshold <= 0.3 + eps:
                        break
                    print(new_marker_rate, neighbor_rate_threshold, distance_rate_threshold)
                elif new_marker_rate > new_marker_rate_range[1]:
                    neighbor_rate_threshold += 0.1
                    distance_rate_threshold -= 0.1
                    neighbor_rate_threshold = np.min((0.9, neighbor_rate_threshold))
                    distance_rate_threshold = np.max((0, distance_rate_threshold))
                    if distance_rate_threshold >= - eps and neighbor_rate_threshold <= 0.9 + eps:
                        break
                    print(new_marker_rate, neighbor_rate_threshold, distance_rate_threshold)
                else:
                    break

            if len(new_markers) == 0:
                print(name, 'failed!!')
                continue # this method failed

            new_markers = np.array(new_markers)
            print('new markers rate: ', new_markers.shape[0] / markers.shape[0])
            markers = new_markers

        original_markers = markers.copy()
        ### build correspondence between the source and target ###
        while True: # until no new marker built
            fm3_reg_target_G = non_rigid_registration(fm3_source_G, fm3_target_G, markers, alpha=0, beta=5, gamma=1000, iter=1000)  # deformation
            new_markers = build_correspondence_v4(fm3_source_G, fm3_reg_target_G, markers, step=2)  # matching
            fm3_target_G = fm3_reg_target_G
            if new_markers.shape[0] <= markers.shape[0]:
                break
            markers = new_markers

        ### scale the target ###
        e1 = mean_link_length(source_G)
        e3 = mean_link_length(deformed_source_G)
        scaled_target_G = target_G.copy()
        scale = e3 / e1
        center = np.mean(scaled_target_G.nodes, axis=0)
        scaled_target_G.nodes = (scaled_target_G.nodes - center) * scale + center

        # ### align the target to the source ###
        # R, t = aligning(deformed_source_G, scaled_target_G, markers)
        # scaled_target_G.nodes = scaled_target_G.nodes.dot(R) + t

        ### register target into the source first and then the deformed source ###
        # reg_target_G = non_rigid_registration(source_G, target_G, marker, alpha=0, beta=1, gamma=1000, iter=1000)
        deformed_target_G = non_rigid_registration(deformed_source_G, scaled_target_G, markers, alpha=500, beta=1, gamma=50, iter=1000)

        ### rescale the deformed target back ###
        center = np.mean(deformed_target_G.nodes, axis=0)
        deformed_target_G.nodes = (deformed_target_G.nodes - center) / scale + center
        target_original_markers = np.array([[marker, marker] for marker in original_markers[:, 1]])
        R, t = aligning(target_G, deformed_target_G, target_original_markers)
        deformed_target_G.nodes = deformed_target_G.nodes.dot(R.T) + t
        # deformed_target_G.nodes = scale(target_Gs[i], reg_target_G)
        
        deformed_target_Gs[name] = {
            "filtered_markers": original_markers,
            "deformed_target_G": deformed_target_G
        }
    return deformed_target_Gs, markers

def modification_transfer(source_G, target_G, markers, intermediate_states=[], inter_res=False):
    # alignment
    raw_target_G = target_G.copy()
    # target_G = Graph(layout(target_G.rawgraph))

    R0, t0 = aligning(intermediate_states[0], target_G, markers)
    align_target_G = target_G.copy()
    align_target_G.nodes = target_G.nodes.dot(R0.T) + t0
    distance_matrix = compute_distance_matrix(align_target_G.nodes, intermediate_states[0].nodes)
    min_dis_sum = np.sum(np.min(distance_matrix, axis=1))
    flip_axis = -1
    for i in range(2):
        _target_G = target_G.copy()
        _target_G.nodes[:, i] = np.mean(_target_G.nodes[:, i]) - (_target_G.nodes[:, i] - np.mean(_target_G.nodes[:, i]))
        R, t = aligning(intermediate_states[0], _target_G, markers)
        _align_target_G = _target_G
        _align_target_G.nodes = _target_G.nodes.dot(R.T) + t
        distance_matrix = compute_distance_matrix(_align_target_G.nodes, intermediate_states[0].nodes)
        _min_dis_sum = np.sum(np.min(distance_matrix, axis=1))
        if _min_dis_sum < min_dis_sum:
            flip_axis = i
            min_dis_sum = _min_dis_sum
            align_target_G = _align_target_G
            R0 = R
            t0 = t

    target_G = align_target_G.copy()

    # deform to the final state through intermediate states
    deformation_target_Gs = []  # every deformations, return to intermediate results
    inter_markers = []  # every matchings, return to intermediate results
    # R = np.eye(2)
    # t = np.zeros((2))
    for intermediate_state in intermediate_states:
        # deformation and matching (target 2 source)
        # until no more correspondece are built
        marker_increasing = True
        while marker_increasing:
            R, t = aligning(source_G, target_G, markers)
            target_G.nodes = target_G.nodes.dot(R.T) + t
            reg_target_G = non_rigid_registration(intermediate_state, target_G, markers, alpha=0, beta=5, gamma=1000, iter=1000)  # deformation
            new_markers = build_correspondence_v4(intermediate_state, reg_target_G, markers, step=1)  # matching
            # new_markers = build_correspondence_v1(intermediate_state, reg_target_G, markers, rate=2)  # matching
            #####
            _markers = new_markers.copy()
            _markers[:, 0] = np.array([source_G.index2id[marker] for marker in _markers[:, 0]])
            _markers[:, 1] = np.array([target_G.index2id[marker] for marker in _markers[:, 1]])
            print(_markers)
            #####
            inter_markers.append(new_markers.copy())
            target_G = reg_target_G
            if new_markers.shape[0] <= markers.shape[0]:
                marker_increasing = False
            markers = new_markers

        deformation_target_G = target_G.copy()
        deformation_target_G.nodes = target_G.nodes
        deformation_target_Gs.append(deformation_target_G)


    R1, t1 = aligning(raw_target_G, deformation_target_Gs[0], np.array([[index, index] for index in target_G.index2id]))
    target_G.nodes = target_G.nodes.dot(R1.T) + t1

    target_G.nodes = scale(raw_target_G, target_G)

    # if flip_axis >= 0:
    #     target_G.nodes[:, flip_axis] = np.mean(target_G.nodes[:, flip_axis]) - (
    #         target_G.nodes[:, flip_axis] - np.mean(target_G.nodes[:, flip_axis]))

    # R, t = aligning(raw_target_G, target_G, np.array([[index, index] for index in target_G.index2id]))
    # target_G.nodes = target_G.nodes.dot(R.T) + t
    # target_G.nodes = (target_G.nodes - t0).dot(np.linalg.inv(R0).T) ############

    if inter_res:
        return target_G, {
            "alignment": align_target_G, # the target graph after alignment
            "deformations": deformation_target_Gs, # for each intermediate state, a deformed target is generated
            "matchings": inter_markers, # markers are built iteratively
        }
    else:
        return target_G

def main(prefix, G, source_G, deformed_source_G, target_Gs, markers):
    fm3_source_G = Graph(layout(source_G.to_networkx()))
    intermediate_states = [fm3_source_G, source_G, deformed_source_G]

    shutil.rmtree(prefix + "result")
    os.mkdir(prefix + "result")

    deformed_targets = [deformed_source_G]
    for i in range(len(target_Gs)):
        target_G = target_Gs[i].copy()
        fm3_target_G = Graph(layout(target_G.to_networkx()))
        # change id2id markers into the index2index markers
        markers[i] = np.array(markers[i])  # [source, target]
        markers[i][:, 0] = np.array([source_G.id2index[str(id)] for id in markers[i][:, 0]])
        markers[i][:, 1] = np.array([target_G.id2index[str(id)] for id in markers[i][:, 1]])

        result = modification_transfer(fm3_source_G, fm3_target_G, markers[i], [fm3_source_G], inter_res=True)
        marker = result[1]['matchings'][-1]

        R, t = aligning(source_G, target_G, marker)
        align_target_G = target_G.copy()
        align_target_G.nodes = align_target_G.nodes.dot(R.T) + t
        align_target = align_target_G.to_networkx()
        # align_target = nx.union(align_target_G.to_networkx(), source_G.to_networkx())

        target_G = align_target_G
        e1 = mean_link_length(source_G)
        e3 = mean_link_length(deformed_source_G)

        center = np.mean(deformed_source_G.nodes, axis=0)
        deformed_source_G.nodes = (deformed_source_G.nodes - center) / e3 * e1 + center

        reg_target_G = non_rigid_registration(deformed_source_G, target_G, marker, alpha=0, beta=1, gamma=1000, iter=1000)

        deformed_target = reg_target_G.to_networkx()
        align_markers = np.array([[target_Gs[i].id2index[target_G.index2id[idx]], reg_target_G.id2index[target_G.index2id[idx]]] for idx in markers[i][:, 1]])
        R, t = aligning(target_Gs[i], reg_target_G, align_markers)
        reg_target_G.nodes = reg_target_G.nodes.dot(R.T) + t
        reg_target_G.nodes = scale(target_Gs[i], reg_target_G)
        # deformed_target = remove_overlap(deformed_target)
        # deformed_target_G = reg_target_G # Graph(deformed_target)
        deformed_targets.append(reg_target_G)

        # deformed_target = reg_target_G.to_networkx()
        target = target_Gs[i].to_networkx()
        for node in target.nodes:
            G.nodes[int(node)]['color'] = ['#436dba']
            target.nodes[node]['color'] = ['#436dba']
            align_target.nodes[node]['color'] = ['#436dba']
            deformed_target.nodes[node]['color'] = ['#436dba']

        save_json_graph(target, prefix + '/result/target' + str(i) + '.json')
        save_json_graph(align_target, prefix + '/result/aligned_target' + str(i) + '.json')
        # save_json_graph(nx.union(nx.relabel_nodes(reg_target_G.to_networkx(), lambda x: str(x) + 's'), deformed_target), prefix + '/result/deformed_target' + str(i) + '.json')
        # save_json_graph(radial_tree_layout(deformed_target), prefix + '/result/deformed_target' + str(i) + '.json')
        save_json_graph(deformed_target, prefix + '/result/deformed_target' + str(i) + '.json')

        inter_deformaed_target_Gs = [result[1]['deformations'][-1], reg_target_G]
        for k in range(len(inter_deformaed_target_Gs)):
            _inter_deformaed_target = inter_deformaed_target_Gs[k].to_networkx()
            # inter_deformaed_target = nx.union(_inter_deformaed_target, intermediate_states[k].to_networkx())
            inter_deformaed_target = _inter_deformaed_target
            for node in _inter_deformaed_target.nodes:
                inter_deformaed_target.nodes[node]['color'] = ['#436dba']
            save_json_graph(inter_deformaed_target, prefix + '/result/deformed_target' + str(i) + '_' + str(k) + '.json')


    for k in range(len(intermediate_states)):
        inter_state = intermediate_states[k].to_networkx()
        for node in inter_state.nodes:
            G.nodes[int(node)]['color'] = ['#f06f6b']
            inter_state.nodes[node]['color'] = ['#f06f6b']
        save_json_graph(inter_state, prefix + '/result/interpolation' + str(k) + '.json')

    save_json_graph(G, prefix + '/result/pos.json')
    G0, G1 = merge(Graph(G), deformed_targets, iter=100, alpha=0, beta=1, gamma=1000)
    save_json_graph(G0.to_networkx(), prefix + '/result/new.json')
    save_json_graph(G1.to_networkx(), prefix + '/result/_new.json')
    return G0.to_networkx()

def main_for_power():
    def modify(source_G, source_nodes):
        V = source_G.nodes
        n = V.shape[0]
        center = np.mean(V, axis=0)
        radius = np.mean(np.sqrt(np.sum((V - center)**2, axis=1)))
        interval = 2.0 * np.pi / n
        deformed_source_G = source_G.copy()
        for i in range(n):
            x =  center[0] + radius * np.sin(interval * i)
            y =  center[1] - radius * np.cos(interval * i)
            id = source_nodes[i]
            index = deformed_source_G.id2index[str(id)]
            deformed_source_G.nodes[index] = np.array([x, y])

        R, t = aligning(source_G, deformed_source_G, np.array([[index, index] for index in source_G.index2id]))
        deformed_source_G.nodes = deformed_source_G.nodes.dot(R.T) + t

        return deformed_source_G

    prefix = './data/power-662-bus/'

    # source_nodes = [462, 575, 589, 588, 477, 476, 466, 574]
    # target_nodes = [
    #     [222, 220, 221, 257, 195, 194, 181, 182, 183, 245, 246],
    #     # [482, 487, 580, 583, 488],
    #     [135, 136, 11, 10, 12, 271, 273, 289, 290, 137],
    #     [28, 30, 228, 306, 60, 59, 61, 317, 31],
    #     # [272, 271, 216, 8, 7, 9, 71, 99, 214, 215, 320],
    #     # [466, 476, 478, 557, 556, 545, 474, 473, 475, 465, 464]
    #     [55, 17, 148, 146, 149, 252, 35, 33, 34, 62, 63, 54, 53],
    #     # [126, 19, 18, 20, 169, 263, 204, 203, 100, 71, 70, 72, 113, 41, 329, 326],
    #     [126, 19, 18, 20, 169, 197, 196, 41, 329, 326]
    # ]
    #
    # markers = [
    #     # [[462, 246], [589, 220], [466, 182], [477, 194]],
    #     # [[462, 488], [589, 482], [466, 583], [477, 487]],
    #     # [[462, 137], [589, 273], [466, 11], [477, 12]],
    #     # [[462, 317], [589, 59], [466, 28], [477, 228]],
    #     # [[462, 71], [589, 7], [466, 215], [477, 272]],
    #     ######################3
    #     [[462, 220], [589, 257], [466, 245], [477, 181]],
    #     # [[462, 583], [589, 488], [466, 580], [477, 482]],
    #     [[462, 271], [589, 10], [466, 289], [477, 136]],
    #     [[462, 317], [589, 60], [466, 28], [477, 228]],
    #     # [[462, 272], [589, 215], [466, 7], [477, 99]],
    #     # [[462, 464], [589, 476], [466, 474], [477, 556]],
    #     [[462, 53], [589, 17], [466, 62], [477, 33]],
    #     # [[462, 326], [589, 19], [466, 70], [477, 100]],
    #     [[462, 326], [589, 19], [466, 196], [477, 169]],
    # ]

    source_nodes = [71, 9, 7, 8, 216, 271, 12, 10, 11, 136, 135, 137, 138, 139, 269, 270, 436, 381, 101]
    source_nodes.reverse()
    # [463, 529, 530, 542, 480, 479, 481, 548, 469, 570, 562, 472, 470, 471, 514, 535, 537, 498, 496, 497, 466, 574, 462, 461]
    target_nodes = [
        [428, 264, 181, 194, 195, 257, 221, 220, 222, 280, 171, 170, 172, 337, 428],
        [265, 328, 288, 49, 32, 344, 424, 425, 565, 564, 419, 250, 417, 382, 347, 427],
        [41, 89, 87, 88, 146, 148, 17, 55, 53, 54, 63, 62, 34, 33, 35, 112, 113],
        [463, 529, 530, 542, 540, 541, 468, 467, 469, 570, 562, 472, 470, 471, 514, 535, 537, 498, 496, 497, 466, 574,
         462, 461],
        [595, 597, 616, 643, 638, 637, 635, 634, 631, 611]
    ]
    markers = [
        [[270, 337], [136, 194], [9, 280]],
        [[270, 565], [136, 382], [9, 49]],
        [[270, 41], [136, 148], [9, 34]],
        [[270, 514], [136, 462], [9, 467]],
        [[270, 616], [136, 637], [9, 611]],
    ]

    G = load_json_graph(prefix + 'graph-with-pos.json')
    # print(nx.shortest_path(G, source=71, target=11))
    # print(nx.shortest_path(G, source=11, target=270))
    # print(nx.shortest_path(G, source=270, target=101))
    source = nx.Graph(G.subgraph(source_nodes))
    source_G = Graph(source)
    deformed_source_G = modify(source_G, source_nodes)

    # target_Gs = []
    # for i in range(len(target_nodes)):
    #     target = nx.Graph(G.subgraph(target_nodes[i]))
    #     target_G = Graph(target)
    #     target_Gs.append(target_G)

    for i in range(len(target_nodes)):
        target = nx.Graph(G.subgraph(target_nodes[i]))
        target_G = Graph(target)
        deformed_target_Gs, markers = generate_G(source_G, deformed_source_G, target_G)
        j = 0
        for name in deformed_target_Gs:
            deformed_target = deformed_target_Gs[name]['deformed_target_G'].to_networkx()
            print(name, 'filter rate:', deformed_target_Gs[name]['filtered_markers'] / np.min((source_G.nodes.shape[0], target_G.nodes.shape[0])))
            save_json_graph(deformed_target, './data/power-662-bus/result/deformed_target' + str(i) + '_' + name + '.json')
            j += 1

    # main(prefix, G, source_G, deformed_source_G, target_Gs, markers)

def main_for_power_compare():
    def modify(source_G, source_nodes):
        V = source_G.nodes
        n = V.shape[0]
        center = np.mean(V, axis=0)
        radius = np.mean(np.sqrt(np.sum((V - center)**2, axis=1)))
        interval = 2.0 * np.pi / n
        deformed_source_G = source_G.copy()
        for i in range(n):
            x =  center[0] + radius * np.sin(interval * i)
            y =  center[1] - radius * np.cos(interval * i)
            id = source_nodes[i]
            index = deformed_source_G.id2index[str(id)]
            deformed_source_G.nodes[index] = np.array([x, y])

        R, t = aligning(source_G, deformed_source_G, np.array([[index, index] for index in source_G.index2id]))
        deformed_source_G.nodes = deformed_source_G.nodes.dot(R.T) + t

        return deformed_source_G

    prefix = './data/power-662-bus/'
    source_nodes = [463, 529, 530, 542, 540, 541, 468, 467, 469, 570, 562, 472, 470, 471, 514, 535, 537, 498, 496, 497, 466, 574, 462, 461]
    target_nodes = [
        # [265, 328, 288, 49, 32, 344, 424, 425, 565, 564, 419, 250, 417, 382, 347, 427],
        [428, 264, 181, 194, 195, 257, 221, 220, 222, 280, 171, 170, 172, 337, 428],
    ]
    markers = [
        # [[514, 265], [462, 344], [467, 250]],
        [[514, 337], [462, 194], [467, 280]],
    ]

    G = load_json_graph(prefix + 'graph-with-pos.json')
    source = nx.Graph(G.subgraph(source_nodes))
    target = nx.Graph(G.subgraph(target_nodes[0]))
    source_G = Graph(source)
    target_G = Graph(target)
    deformed_source_G = modify(source_G, source_nodes)
    H = nx.union(source, target)

    source_node_link_data = json.dumps(json_graph.node_link_data(source_G.to_networkx()))
    target_node_link_data = json.dumps(json_graph.node_link_data(target_G.to_networkx()))
    M = fgm(source_node_link_data, target_node_link_data)
    target_Gs = [target_G]


    print(M)
    for i in range(len(names)):
        name = names[i]
        markers.append(M[name].tolist())

        # marker = fgm(source_node_link_data, target_node_link_data)
        # correspondence = {}
        # for name in marker:
        #     for tuple in marker[name]:
        #         sid = tuple[0]
        #         tid = tuple[1]
        #         if sid not in correspondence:
        #             correspondence[sid] = []
        #         correspondence[sid].append(tid)
        # marker = []
        # for sid in correspondence:
        #     unique_elements, counts_elements = np.unique(correspondence[sid], return_counts=True)
        #     if counts_elements[0] / len(correspondence[sid]) >= 0.3:
        #         tid = unique_elements[0]
        #         marker.append([sid, tid])
        # markers.append(marker)
        target_Gs.append(target_G)

    # markers = [[[source_nodes[i], target_nodes[k][i]] for i in [0, 1, -1]] for k in range(len(target_nodes))]
    main(prefix, H, source_G, deformed_source_G, target_Gs, markers)

def main_for_price():
    def modify(graph):
        id2pos = {"3":{"x":603.1899784256054,"y":621.8612844808738},"112":{"x":356.4605163829269,"y":224.3324684324944},"113":{"x":916.455318727647,"y":220.90882652619246},"114":{"x":564.9949590994031,"y":224.54044889591077},"116":{"x":1195.612292042225,"y":228.0732869394473},"117":{"x":186.34624221898628,"y":225.09470045417606},"118":{"x":2.162575231935051,"y":227.79561664404724},"119":{"x":509.69671268929164,"y":226.78346403098794},"120":{"x":224.6607374895325,"y":224.32525359195105},"121":{"x":856.7573107496587,"y":220.154308484966},"122":{"x":140.90001938807922,"y":224.2806651348709},"123":{"x":1078.9793578833169,"y":222.62146219118296},"124":{"x":98.05544990656597,"y":225.01550221312863},"125":{"x":49.863875058267354,"y":227.38734494693028},"126":{"x":454.08349249433826,"y":228.25431797274945},"127":{"x":268.1137708989056,"y":222.25151288894756},"128":{"x":740.3952651835474,"y":223.2924341114691},"129":{"x":972.2371945861883,"y":220.09119409942372},"130":{"x":406.1311459179843,"y":227.55831067953153},"131":{"x":309.80146606879885,"y":223.68639988119645},"132":{"x":683.6194450059195,"y":223.91754703671774},"133":{"x":801.1533597066212,"y":222.72593506231},"134":{"x":617.6553129524744,"y":225.95272608769187},"362":{"x":915.0070044592583,"y":11.060573309627614},"363":{"x":564.6118547791615,"y":9.324568821493529},"380":{"x":1200.6757281873797,"y":-1.4431619894644427},"381":{"x":1151.6275714986987,"y":-0.002693705474456465},"382":{"x":1254.642570446792,"y":-1.0440150649500595},"383":{"x":0.11603971668654367,"y":15.510933639142479},"384":{"x":-76.36233432473796,"y":17.886163099122598},"385":{"x":75.48321281098274,"y":15.629436054402106},"386":{"x":144.5754835675716,"y":14},"387":{"x":1036.0447287112702,"y":0.5539132981597845},"388":{"x":1104.9388739238584,"y":0.6869155216012075},"389":{"x":267.3739382964066,"y":12.554540316891973},"390":{"x":737.3954054377375,"y":7.007825358782611},"391":{"x":406.7485159869151,"y":10.577055239525293},"618":{"x":1200.1686309128486,"y":-183},"619":{"x":1153.2356229142347,"y":-180.94739642846832},"620":{"x":1258.5623863012906,"y":-182.19348478035784},"621":{"x":-0.3241645594162037,"y":-161.4108787750034},"622":{"x":1038.3241645594162,"y":-182.05755371941737}}
        for id in id2pos:
            graph.nodes[int(id)]['x'] = id2pos[id]['x'] / 4
            graph.nodes[int(id)]['y'] = id2pos[id]['y'] / 4
        return graph

    prefix = './data/price/'
    G = load_json_graph(prefix + 'graph-with-pos.json')

    # source_nodes = [63, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 561, 562, 563, 564, 565, 566, 567, 568,
    #      569, 570, 571, 572, 751, 752, 753, 868, 869]
    # target_nodes = [
    #     [145,406,407,408,409,410,411,412,413,414,415,416,417,667,668,669,670,671,672,673,674,675,676,833,834,835,836,980],
    #     [365,604,605,606,607,608,609,610,611,612,758,759,760,761,762,763,764,870,871],
    #     [287,533,534,535,536,537,538,539,540,541,542,543,544,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,857,858,859,860,861,862,863,864,865]
    # ]
    # markers = [
    #     [[63, 145], [304, 408], [305, 406], [307, 410]],
    #     [[63, 365], [304, 604], [305, 611], [307, 607]],
    #     [[63, 287], [304, 536], [305, 533], [307, 535]],
    # ]

    # source_nodes = [3,112,113,114,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,362,363,380,381,382,383,384,385,386,387,388,389,390,391,618,619,620,621,622]
    # target_nodes = [
    #     [115,364,366,367,368,369,370,371,372,373,374,375,376,377,378,379,603,613,614,615,616,617,765,872],
    #     [365,604,605,606,607,608,609,610,611,612,758,759,760,761,762,763,764,870,871],
    #     [5,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,418,419,420,421,422,423,424,677,678],
    #     [111,354,355,356,357,358,359,360,361,597,598,599,600,601,602,757],
    #     # [2,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,351,352,353,595,596],
    #     # [145,406,407,408,409,410,411,412,413,414,415,416,417,667,668,669,670,671,672,673,674,675,676,833,834,835,836,980],
    #     # [928,966,967,968,969,970,971,972,973,974,975,976,977,978,979,990,991,992,997],
    #     # [4,136,137,138,139,140,141,142,143,144,399,400,401,402,403,404,405,665,666],
    #     # [12,207,210,211,212,213,214,215,216,217,218,219,220,481,482,483,484,485,486,487,488,489,707,708,709,710,711,846,847,848,995],
    #     # [63,303,304,305,306,307,308,309,310,311,312,313,314,315,561,562,563,564,565,566,567,568,569,570,571,572,751,752,753,868,869],
    # ]
    # markers = [
    #     [[3, 115], [118, 368]],
    #     [[3, 365], [118, 604]],
    #     [[3, 5], [118, 151]],
    #     [[3, 111], [118, 354]],
    #     # [[3, 2], [118, 96]],
    #     # [[3, 145], [118, 406]],
    #     # [[3, 928], [118, 966]],
    #     # [[3, 4], [118, 136]],
    #     # [[3, 12], [118, 211]],
    #     # [[3, 63], [118, 304]],
    #
    #     # [[3, 115]],
    #     # [[3, 365]],
    #     # [[3, 5]],
    #     # [[3, 111]],
    #     # [[3, 2]],
    #     # [[3, 145]],
    #     # [[3, 928]],
    #     # [[3, 4]],
    #     # [[3, 12]],
    #     # [[3, 63], [118, 304]]
    # ]

    source_nodes = [3,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,757,758,759,760,761,762,763,764,765,870,871,872]
    target_nodes = [
        [59, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 941],
        [63, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 751, 752, 753, 868, 869],
        [5, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 833, 834, 835, 836, 937],
        [393, 635, 636, 637, 638, 639, 640, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 947, 948, 949, 950, 951, 952],
        [12, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 844, 845, 846, 847, 848, 938, 939],
        [4, 136, 137, 138, 139, 140, 141, 142, 143, 144, 399, 400, 401, 402, 403, 404, 405, 665, 666],
    ]
    markers = [
        [[3, 287], [115, 59]],
        [[3, 63], [115, 304]],
        [[3, 5], [115, 145]],
        [[3, 636], [115, 393]],#, [354, 790]],
        [[3, 12], [115, 208]],#, [354, 209]],
        [[3, 4], [115, 136]],
    ]

    source = nx.Graph(G.subgraph(source_nodes))
    source_G = Graph(source)
    # deformed_source_G = Graph(modify(source.copy()))
    deformed_source_G = Graph(layout(source))
    ####
    r = 0
    c = 0
    center = source_G.nodes[source_G.id2index['3']]
    for node in source.neighbors(3):
        if node == 115:
            continue
        r += np.linalg.norm(source_G.nodes[source_G.id2index[str(node)]] - center)
        c += 1
    r /= c
    center = deformed_source_G.nodes[deformed_source_G.id2index['3']]
    for node in source.nodes:
        if node != 3:
            rate = nx.shortest_path_length(source, node, 3)
            index = deformed_source_G.id2index[str(node)]
            pos = deformed_source_G.nodes[index]
            v = pos-center
            deformed_source_G.nodes[index] = center + v / np.linalg.norm(v) * r * rate

    deformed_source_G = Graph(remove_overlap(deformed_source_G.to_networkx()))
    ####
    # deformed_source_G = Graph(radial_tree_layout(source))

    target_Gs = []
    for i in range(len(target_nodes)):
        target = nx.Graph(G.subgraph(target_nodes[i]))
        target_G = Graph(target)
        target_Gs.append(target_G)

    # R, t = aligning(source_G, deformed_source_G, np.array([[source_G.id2index[id], source_G.id2index[id]] for id in ['3', '118']]))
    # deformed_source_G.nodes = deformed_source_G.nodes.dot(R.T) + t

    main(prefix, G, source_G, deformed_source_G, target_Gs, markers)

def main_for_finan():
    all_nodes = [45135, 45136, 45138, 45139, 45140, 45266, 45267, 45317, 45318, 45319, 45320, 45321, 45362, 45363,
                 45364, 45365, 45366, 45395, 45396, 45397, 45398, 45422, 45423, 45424, 45425, 45449, 45450, 45451,
                 45452, 45476, 45477, 45478, 45479, 45503, 45504, 45505, 45506, 45530, 45531, 45532, 45533, 47048,
                 47050, 47052, 47054, 47056, 47058, 47064, 47065, 47066, 47067, 47068, 47069, 47070, 47071, 47072,
                 47073, 47074, 47097, 47098, 47099, 47100, 47101, 47102, 47103, 47104, 47105, 47106, 47107, 47108,
                 47109, 47110, 47111, 47112, 47113, 47114, 47115, 47116, 47117, 47118, 47119, 47120, 47121, 47122,
                 47123, 47124, 47125, 47126, 47127, 47128, 47129, 47130, 47131, 47132, 47133, 47134, 47135, 47136,
                 47137, 47138, 47139, 47140, 47141, 47142, 47143, 47144, 47145, 47146, 47147, 47148, 47149, 47150,
                 47151, 47152, 47153, 47154, 47155, 47156, 47157, 47158, 47159, 47160, 47161, 47162, 47163, 47164,
                 47165, 47166, 47167, 47168, 47169, 47170, 47171, 47172, 47173, 47174, 47175, 47176, 47177, 47178,
                 47179, 47180, 47181, 47182, 47183, 47184, 47185, 47186, 47187, 47188, 47189, 47190, 47191, 47192,
                 47193, 47194, 47195, 47196, 47197, 47198, 47199, 47203, 47204, 47205, 47206, 47207, 47208, 47209,
                 47210, 47211, 47212, 47213, 47214, 47215, 47216, 47217, 47218, 47219, 47220, 47243, 47244, 47245,
                 47246, 47247, 47248, 47249, 47250, 47251, 47252, 47253, 47254, 47255, 47256, 47257, 47258, 47259,
                 47260, 47261, 47262, 47263, 47264, 47265, 47266, 47267, 47268, 47269, 47270, 47271, 47272, 47273,
                 47274, 47275, 47276, 47277, 47278, 47279, 47280, 47281, 47282, 47283, 47284, 47285, 47286, 47287,
                 47288, 47289, 47290, 47291, 47292, 47293, 47294, 47295, 47296, 47297, 47298, 47299, 47300, 47301,
                 47302, 47303, 47304, 47305, 47306, 47307, 47308, 47309, 47310, 47311, 47312, 47313, 47314, 47315,
                 47316, 47317, 47318, 47319, 47320, 47321, 47322, 47323, 47324, 47325, 47326, 47327, 47328, 47329,
                 47330, 47331, 47332, 47333, 47334, 47335, 47336, 47337, 47338, 47339, 47340, 47341, 47342, 47343,
                 47344, 47345, 47346, 47347, 47348, 47349, 47350, 47351, 47352, 47353, 47354, 47355, 47356, 47357,
                 47358, 47359, 47360, 47361, 47362, 47363, 47364, 47365, 47366, 47367, 47368, 47369, 47370, 47371,
                 47372, 47373, 47374, 47375, 47376, 47377, 47378, 47379, 47380, 47381, 47382, 47383, 47384, 47385,
                 47386, 47387, 47388, 47389, 47390, 47391, 47392, 47393, 47394, 47395, 47398, 47399, 47400, 47401,
                 47402, 47403, 47404, 47405, 47406, 47407, 47408, 47409, 47410, 47412, 47413, 47414, 47415, 47416,
                 47417, 47418, 47419, 47420, 47421, 47422, 47423, 47424, 47425, 47426, 47427, 47428, 47429, 47430,
                 47431, 47432, 47433, 47434, 47435, 47436, 47437, 47438, 47439, 47440, 47441, 47442, 47443, 47444,
                 47445, 47446, 47447, 47448, 47449, 47450, 47451, 47452, 47453, 47454, 47455, 47456, 47477, 47478,
                 47479, 47480, 47481, 47482, 47483, 47484, 47485, 47486, 47487, 47488, 47489, 47490, 47491, 47492,
                 47493, 47494, 47495, 47496, 47497, 47498, 47499, 47500, 47501, 47502, 47503, 47504, 47505, 47506,
                 47507, 47508, 47509, 47510, 47511, 47512, 47513, 47514, 47515, 47516, 47517, 47518, 47519, 47520,
                 47521, 47522, 47523, 47524, 47525, 47526, 47527, 47528, 47529, 47530, 47531, 47532, 47533, 47534,
                 47535, 47536, 47537, 47538, 47539, 47540, 47541, 47542, 47544, 47545, 47546, 47548, 47549, 47550,
                 47552, 47553, 47554, 47556, 47557, 47558, 47560, 47561, 47562, 47564, 47565, 47566, 47567, 47568,
                 47569, 47570, 47571, 47572, 47573, 47574, 47575, 47576, 47577, 47578, 47579, 47580, 47581, 47582,
                 47583, 47584, 47585, 47586, 47587, 47588, 47589, 47590, 47591, 47592, 47593, 47594, 47595, 47596,
                 47597, 47598, 47599, 47600, 47601, 47604, 47605, 47606, 47607, 47608, 47609, 47610, 47611, 47612,
                 47613, 47614, 47615, 47616, 47617, 47618, 47619, 47620, 47621, 47622, 47623, 47624, 47625, 47626,
                 47627, 47628, 47629, 47631, 47632, 47633, 47634, 47635, 47636, 47637, 47638, 47639, 47640, 47641,
                 47642, 47643, 47644, 47645, 47646, 47647, 47648, 47649, 47650, 47651, 47652, 47658, 47659, 47660,
                 47661, 47662, 47663, 47664, 47665, 47666, 47667, 47668, 47669, 47670, 47671, 47672, 47673, 47674,
                 47676, 47677, 47678, 47679, 47680, 47681, 47682, 47683, 47684, 47685, 47686, 47687, 47688, 47689,
                 47690, 47691, 47692, 47693, 47694, 47695, 47696, 47697, 47703, 47704, 47705, 47706, 47707, 47708,
                 47709, 47710, 47711, 47712, 47713, 47714, 47715, 47716, 47717, 47718, 47719, 47720, 47721, 47722,
                 47723, 47724, 47725, 47726, 47727, 47728, 47729, 47730, 47735, 47736, 47737, 47738, 47739, 47740,
                 47741, 47742, 47743, 47744, 47745, 47746, 47747, 47748, 47749, 47750, 47751, 47752, 47753, 47754,
                 47755, 47756, 47757, 47762, 47763, 47764, 47765, 47766, 47767, 47768, 47769, 47770, 47771, 47772,
                 47773, 47774, 47775, 47776, 47777, 47778, 47779, 47780, 47781, 47782, 47783, 47784, 47789, 47790,
                 47791, 47792, 47793, 47794, 47795, 47796, 47797, 47798, 47799, 47800, 47801, 47802, 47803, 47804,
                 47805, 47806, 47807, 47808, 47809, 47810, 47811, 47816, 47817, 47818, 47819, 47820, 47821, 47822,
                 47823, 47824, 47825, 47826, 47827, 47828, 47829, 47830, 47831, 47832, 47833, 47834, 47835, 47836,
                 47837, 47838, 47843, 47844, 47845, 47846, 47847, 47848, 47849, 47850, 47851, 47852, 47853, 47854,
                 47855, 47856, 47857, 47858, 47859, 47860, 47861, 47862, 47863, 47864, 47865, 47870, 47871, 47872,
                 47873, 47874, 47875, 47876, 47877, 47878, 47879, 47880, 47881, 47882, 47883, 47884, 47885, 47886,
                 47887, 47888, 47889, 47890, 47891, 47892, 47893, 47894, 47895, 47896, 47897, 47898, 47899, 47900,
                 47901, 47902, 47903, 47904, 47905, 47906, 47907, 47908, 47909, 47910, 47911, 47912, 47913, 47914,
                 47915, 47916, 47917, 47918, 47919, 47920, 47921, 47922, 47923, 47924, 47925, 47926, 47927, 47928,
                 47929, 47930, 47931, 47932, 47933, 47934, 47935, 47936, 47937, 47938, 47939, 47940, 47941, 47942,
                 47943, 47944, 47945, 47946, 47947, 47948, 47949, 47950, 47951, 47952, 47953, 47954, 47955, 47956,
                 47957, 47958, 47959, 47960, 47961, 47962, 47963, 47964, 47965, 47966, 47967, 47968, 47969, 47970,
                 47971, 47972, 47973, 47974, 47975, 47976, 47977, 47978, 47979, 47980, 47981, 47982, 47983, 47984,
                 47985, 47986, 47987, 47988, 47989, 47990, 47991, 47992, 47993, 47994, 47995, 47996, 47997, 47998,
                 47999, 48000, 48001, 48002, 48003, 48004, 48005, 48006, 48007, 48008, 48009, 48010, 48011, 48012,
                 48013, 48014, 48015, 48016, 48017, 48018, 48019, 48020, 48021, 48022, 48023, 48024, 48025, 48026,
                 48027, 48028, 48029, 48030, 48031, 48032, 48033, 48034, 48035, 48036, 48037, 48038, 48039, 48040,
                 48041, 48042, 48043, 48044, 48045, 48046, 48047, 48048, 48049, 48050, 48051, 48052, 48053, 48054,
                 48055, 48056, 48057, 48058, 48059, 48060, 48061, 48062, 48063, 48064, 48065, 48066, 48067, 48068,
                 48069, 48070, 48071, 48072, 48073, 48074, 48075, 48076, 48077, 48078, 48079, 48080, 48081, 48082,
                 48083, 48084, 48085, 48086, 48087, 48088, 48089, 48090, 48091, 48092, 48093, 48094, 48095, 48096,
                 48097, 48098, 48099, 48100, 48101, 48102, 48103, 48104, 48105, 48106, 48107, 48108, 48109, 48110,
                 48111, 48112, 48113, 48114, 48115, 48116, 48117, 48118, 48119, 48120, 48121, 48122, 48123, 48124,
                 48125, 48126, 48127, 48128, 48129, 48130, 48131, 48132, 48133, 48134, 48135, 48136, 48137, 48138,
                 48139, 48140, 48141, 48142, 48143, 48144, 48145, 48146, 48147, 48148, 48149, 48150, 48151, 48152,
                 48153, 48154, 48155, 48156, 48157, 48158, 48159, 48160, 48161, 48162, 48163, 48164, 48165, 48166,
                 48167, 48168, 48169, 48170, 48171, 48172, 48173, 48174, 48175, 48176, 48177, 48178, 48179, 48180,
                 48181, 48182, 48183, 48184, 48185, 48186, 48187, 48188, 48189, 48190, 48191, 48192, 48193, 48194,
                 48195, 48196, 48197, 48198, 48199, 48200, 48201, 48202, 48203, 48204, 48205, 48206, 48207, 48208,
                 48209, 48210, 48211, 48212, 48213, 48214, 48215, 48216, 48217, 48218, 48219, 48220, 48221, 48222,
                 48223, 48224, 48225, 48226, 48227, 48228, 48229, 48230, 48231, 48232, 48233, 48234, 48235, 48236,
                 48237, 48238, 48239, 48240, 48241, 48242, 48243, 48244, 48245, 48246, 48247, 48248, 48249, 48250,
                 48251, 48252, 48253, 48254, 48255, 48256, 48257, 48258, 48259, 48260, 48261, 48262, 48263, 48264,
                 48265, 48266, 48267, 48268, 48269, 48270, 48271, 48272, 48273, 48274, 48275, 48276, 48277, 48278,
                 48279, 48280, 48281, 48282, 48283, 48284, 48285, 48286, 48287, 48288, 48289, 48290, 48291, 48292,
                 48293, 48294, 48295, 48296, 48297, 48298, 48299, 48300, 48301, 48302, 48303, 48304, 48305, 48306,
                 48307, 48308, 48309, 48310, 48311, 48312, 48313, 48314, 48315, 48316, 48317, 48318, 48319, 48320,
                 48321, 48322, 48323, 48324, 48325, 48326, 48327, 48328, 48329, 48330, 48331, 48332, 48333, 48334,
                 48335, 48336, 48337, 48338, 48339, 48340, 48341, 48342, 48343, 48344, 48345, 48346, 48347, 48348,
                 48349, 48350, 48351, 48352, 48353, 48354, 48355, 48356, 48357, 48358, 48359, 48360, 48361, 48362,
                 48363, 48364, 48365, 48366, 48367, 48368, 48369, 48370, 48371, 48372, 48373, 48374, 48375, 48376,
                 48377, 48378, 48379, 48380, 48381, 48382, 48383, 48384, 48385, 48386, 48387, 48388, 48389, 48390,
                 48391, 48392, 48393, 48394, 48395, 48396, 48397, 48398, 48399, 48400, 48401, 48402, 48403, 48404,
                 48405, 48406, 48407, 48408, 48409, 48410, 48411, 48412, 48413, 48414, 48415, 48416, 48417, 48418,
                 48419, 48420, 48421, 48422, 48423, 48424, 48425, 48426, 48427, 48428, 48429, 48430, 48431, 48432,
                 48433, 48434, 48435, 48436, 48437, 48438, 48439, 48440, 48441, 48442, 48443, 48444, 48445, 48446,
                 48447, 48448, 48449, 48450, 48451, 48452, 48453, 48454, 48455, 48456, 48457, 48458, 48459, 48460,
                 48461, 48462, 48463, 48464, 48465, 48466, 48467, 48468, 48469, 48470, 48471, 48472, 48473, 48474,
                 48475, 48476, 48477, 48478, 48479, 48480, 48481, 48482, 48483, 48484, 48485, 48486, 48487, 48488,
                 48489, 48490, 48491, 48492, 48493, 48494, 48495, 48496, 48497, 48498, 48499, 48500, 48501, 48502,
                 48503, 48504, 48505, 48506, 48507, 48508, 48509, 48510, 48511, 48512, 48513, 48514, 48515, 48516,
                 48517, 48518, 48519, 48520, 48521, 48522, 48523, 48524, 48525, 48526, 48527, 48528, 48529, 48530,
                 48531, 48532, 48533, 48534, 48535, 48536, 48537, 48538, 48539, 48540, 48541, 48542, 48543, 48544,
                 48545, 48546, 48547, 48548, 48549, 48550, 48551, 48552, 48553, 48554, 48555, 48556, 48557, 48558,
                 48559, 48560, 48561, 48562, 48563, 48564, 48565, 48566, 48567, 48568, 48569, 48570, 48571, 48572,
                 48573, 48574, 48575, 48576, 48577, 48578, 48579, 48580, 48581, 48582, 48583, 48584, 48585, 48586,
                 48587, 48588, 48589, 48590, 48591, 48592, 48593, 48594, 48595, 48596, 48597, 48598, 48599, 48600,
                 48601, 48602, 48603, 48604, 48605, 48606, 48607, 48608, 48609, 48610, 48611, 48612, 48613, 48614,
                 48615, 48616, 48617, 48618, 48619, 48620, 48621, 48622, 48623, 48624, 48625, 48626, 48627, 48628,
                 48629, 48630, 48631, 48632, 48633, 48634, 48635, 48636, 48637, 48638, 48639, 48640, 48641, 48642,
                 48643, 48644, 48645, 48646, 48647, 48648, 48649, 48650, 48651, 48652, 48653, 48654, 48655, 48656,
                 48657, 48658, 48659, 48660, 48661, 48662, 48663, 48664, 48665, 48666, 48667, 48668, 48669, 48670,
                 48671, 48672, 48673, 48674, 48675, 48676, 48677, 48678, 48679, 48680, 48681, 48682, 48683, 48684,
                 48685, 48686, 48687, 48688, 48689, 48690, 48691, 48692, 48693, 48694, 48695, 48696, 48697, 48698,
                 48699, 48700, 48701, 48702, 48703, 48704, 48705, 48706, 48707, 48708, 48709, 48710, 48711, 48712,
                 48713, 48714, 48715, 48716, 48717, 48718, 48719, 48720, 48721, 48722, 48723, 48724, 48725, 48726,
                 48727, 48728, 48729, 48730, 48731, 48732, 48733, 48734, 48735, 48736, 48737, 48738, 48739, 48740,
                 48741, 48742, 48743, 48744, 48745, 48746, 48747, 48748, 48749, 48750, 48751, 48752, 48753, 48754,
                 48755, 48756, 48757, 48758, 48759, 48760, 48761, 48762, 48763, 48764, 48765, 48766, 48767, 48768,
                 48769, 48770, 48771, 48772, 48773, 48774, 48775, 48776, 48777, 48778, 48779, 48780, 48781, 48782,
                 48783, 48784, 48785, 48786, 48787, 48788, 48789, 48790, 48791, 48792, 48793, 48794, 48795, 48796,
                 48797, 48798, 48799, 48800, 48801, 48802, 48803, 48804, 48805, 48806, 48807, 48808, 48809, 48810,
                 48811, 48812, 48813, 48814, 48815, 48816, 48817, 48818, 48819, 48820, 48821, 48822, 48823, 48824,
                 48825, 48826, 48827, 48828, 48829, 48830, 48831, 48832, 48833, 48834, 48835, 48836, 48837, 48838,
                 48839, 48840, 48841, 48842, 48843, 48844, 48845, 48846, 48847, 48848, 48849, 48850, 48851, 48852,
                 48853, 48854, 48855, 48856, 48857, 48858, 48859, 48860, 48861, 48862, 48863, 48864, 48865, 48866,
                 48867, 48868, 48869, 48870, 48871, 48872, 48873, 48874, 48875, 48876, 48877, 48878, 48879, 48880,
                 48881, 48882, 48883, 48884, 48885, 48886, 48887, 48888, 48889, 48890, 48891, 48892, 48893, 48894,
                 48895, 48896, 48897, 48898, 48899, 48900, 48901, 48902, 48903, 48904, 48905, 48906, 48907, 48908,
                 48909, 48910, 48911, 48912, 48913, 48914, 48915, 48916, 48917, 48918, 48919, 48920, 48921, 48922,
                 48923, 48924, 48925, 48926, 48927, 48928, 48929, 48930, 48931, 48932, 48933, 48934, 48935, 48936,
                 48937, 48938, 48939, 48940, 48941, 48942, 48943, 48944, 48945, 48946, 48947, 48948, 48949, 48950,
                 48951, 48952, 48953, 48954, 48955, 48956, 48957, 48958, 48959, 48960, 48961, 48962, 48963, 48964,
                 48965, 48966, 48967, 48968, 48969, 48970, 48971, 48972, 48973, 48974, 48975, 48976, 48977, 48978,
                 48979, 48980, 48981, 48982, 48983, 48984, 48985, 48986, 48987, 48988, 48989, 48990, 48991, 48992,
                 48993, 48994, 48995, 48996, 48997, 48998, 48999, 49000, 49001, 49002, 49003, 49004, 49005, 49006,
                 49007, 49008, 49009, 49010, 49011, 49012, 49013, 49014, 49015, 49016, 49017, 49018, 49019, 49020,
                 49021, 49022, 49023, 49024, 49025, 49026, 49027, 49028, 49029, 49030, 49031, 49032, 49033, 49034,
                 49035, 49036, 49037, 49038, 49039, 49040, 49041, 49042, 49043, 49044, 49045, 49046, 49047, 49048,
                 49049, 49050, 49051, 49052, 49053, 49054, 49055, 49056, 49057, 49058, 49059, 49060, 49061, 49062,
                 49063, 49064, 49065, 49066, 49067, 49068, 49069, 49070, 49071, 49072, 49073, 49074, 49075, 49076,
                 49077, 49078, 49079, 49080, 49081, 49082, 49083, 49084, 49085, 49086, 49087, 49088, 49089, 49090,
                 49091, 49092, 49093, 49094, 49095, 49096, 49097, 49098, 49099, 49100, 49101, 49102, 49103, 49104,
                 49105, 49106, 49107, 49108, 49109, 49110, 49111, 49112, 49113, 49114, 49115, 49116, 49117, 49118,
                 49119, 49120, 49121, 49122, 49123, 49124, 49125, 49126, 49127, 49128, 49129, 49130, 49131, 49132,
                 49133, 49134, 49135, 49136, 49137, 49138, 49139, 49140, 49141, 49142, 49143, 49144, 49145, 49146,
                 49147, 49148, 49149, 49150, 49151, 49152, 49153, 49154, 49155, 49156, 49157, 49158, 49159, 49160,
                 49161, 49162, 49163, 49164, 49165, 49166, 49167, 49168, 49169, 49170, 49171, 49172, 49173, 49174,
                 49175, 49176, 49177, 49178, 49179, 49180, 49181, 49182, 49183, 49184, 49185, 49186, 49187, 49188,
                 49189, 49190, 49191, 49192, 49193, 49194, 49195, 49196, 49197, 49198, 49199, 49200, 49201, 49202,
                 49203, 49204, 49205, 49206, 49207, 49208, 49209, 49210, 49211, 49212, 49213, 49214, 49215, 49216,
                 49217, 49218, 49219, 49220, 49221, 49222, 49223, 49224, 49225, 49226, 49227, 49228, 49229, 49230,
                 49231, 49232, 49233, 49234, 49235, 49236, 49237, 49238, 49239, 49240, 49241, 49242, 49243, 49244,
                 49245, 49246, 49247, 49248, 49249, 49250, 49251, 49252, 49253, 49254, 49255, 49256, 49257, 49258,
                 49259, 49260, 49261, 49262, 49263, 49264, 49265, 49266, 49267, 49268, 49269, 49270, 49271, 49272,
                 49273, 49274, 49275, 49276, 49277, 49278, 49279, 49280, 49281, 49282, 49283, 49284, 49285, 49286,
                 49287, 49288, 49289, 49290, 49291, 49292, 49293, 49294, 49295, 49296, 49297, 49298, 49299, 49300,
                 49301, 49302, 49303, 49304, 49305, 49306, 49307, 49308, 49309, 49310, 49311, 49312, 49313, 49314,
                 49315, 49316, 49317, 49318, 49319, 49320, 49321, 49322, 49323, 49324, 49325, 49326, 49327, 49328,
                 49329, 49330, 49331, 49332, 49333, 49334, 49335, 49336, 49337, 49338, 49339, 49340, 49341, 49342,
                 49343, 49344, 49345, 49346, 49347, 49348, 49349, 49350, 49351, 49352, 49353, 49354, 49355, 49356,
                 49357, 49358, 49359, 49360, 49361, 49362, 49363, 49364, 49365, 49366, 49367, 49368, 49369, 49370,
                 49371, 49372, 49373, 49374, 49375, 49376, 49377, 49378, 49379, 49380, 49381, 49382, 49383, 49385,
                 49387, 49389, 49391, 49393, 49395, 49396, 49397, 49398, 49399, 49411, 49412, 49413, 49414, 49415,
                 49416, 49417, 49418, 49419, 49420, 49421, 49422, 49423, 49424, 49425, 49426, 49427, 49428, 49429,
                 49430, 49431, 49432, 49536, 49537, 49538, 49557, 49558, 49559, 49560, 49561, 49562, 49563, 49564,
                 49565, 49566, 49567, 49568, 49569, 49570, 49571, 49572, 49573, 49574, 49575, 49576, 49577, 49578]

    # source_nodes = [48761, 48775, 48776, 48777, 48814, 48815, 48816, 48830, 48831, 48832, 48845, 48856, 48867, 48878, 48889, 48900,
    #      48906, 48918, 48919, 48920, 48957, 48958, 48959, 48973, 48974, 48975, 48990, 49001, 49012, 49023, 49034, 49045,
    #      49051, 49055, 49056, 49057, 49073, 49074, 49076, 49077, 49078, 49079, 49080, 49081, 49082, 49083, 49084, 49085,
    #      49086, 49087, 49088, 49089, 49090, 49091, 49092, 49093, 49094, 49095]
    # target_nodes = [
    #     [48615, 48629, 48630, 48631, 48668, 48669, 48670, 48684, 48685, 48686, 48699, 48710, 48721, 48732, 48743, 48754,
    #      48760, 48772, 48773, 48774, 48811, 48812, 48813, 48827, 48828, 48829, 48844, 48855, 48866, 48877, 48888, 48899,
    #      48905, 48909, 48910, 48911, 48928, 48929, 48932, 48933, 48934, 48935, 48936, 48937, 48938, 48939, 48940, 48941,
    #      48942, 48943, 48944, 48945, 48946, 48947, 48948, 48949, 48950, 48951],
    #     [49053, 49067, 49068, 49069, 49104, 49105, 49106, 49118, 49119, 49120, 49130, 49138, 49146, 49154, 49281, 49282,
    #      49162, 49170, 49174, 49182, 49183, 49184, 49212, 49213, 49214, 49220, 49221, 49222, 49228, 49232, 49236, 49240,
    #      49244, 49248, 49250, 49252, 49253, 49254, 49261, 49262, 49263, 49264, 49265, 49266, 49267, 49268, 49269, 49270,
    #      49271, 49272, 49273, 49274, 49275, 49276, 49277, 49278, 49279, 49280],
    #     [49215, 49229, 49233, 49241, 49245, 49249, 49251, 49255, 49256, 49257, 49283, 49284, 49285, 49289, 49290, 49291,
    #      49295, 49297, 49299, 49301, 49303, 49305, 49307, 49309, 49310, 49311, 49315, 49316, 49317, 49318, 49319, 49320,
    #      49321, 49322, 49323, 49324, 49325, 49326, 49327, 49328, 49329, 49330, 49331, 49332, 49333, 49334, 49335, 49336],
    #     [47374, 47398, 47399, 47400, 47477, 47478, 47479, 47566, 47569, 47572, 47575, 47578, 47581, 47584, 47585, 47586,
    #      47587, 47588, 47591, 47595, 47658, 47659, 47660, 47703, 47704, 47705, 47737, 47764, 47791, 47818, 47845, 47872,
    #      47883, 49536, 49537, 49538, 49557, 49558, 49559, 49560, 49561, 49562, 49563, 49564, 49565, 49566, 49567, 49568,
    #      49569, 49570, 49571, 49572, 49573, 49574, 49575, 49576, 49577, 49578],
    #     [48031, 48045, 48046, 48047, 48084, 48085, 48086, 48100, 48101, 48102, 48115, 48126, 48137, 48148, 48159, 48170,
    #      48176, 48188, 48189, 48190, 48227, 48228, 48229, 48243, 48244, 48245, 48260, 48271, 48282, 48293, 48304, 48315,
    #      48321, 48325, 48326, 48327, 48344, 48345, 48348, 48349, 48350, 48351, 48352, 48353, 48354, 48355, 48356, 48357,
    #      48358, 48359, 48360, 48361, 48362, 48363, 48364, 48365, 48366, 48367],
    #     [47642, 47687, 47725, 47752, 47779, 47806, 47833, 47880, 47896, 47897, 47898, 47935, 47936, 47937, 47951, 47952,
    #      47953, 47968, 47979, 47990, 48001, 48012, 48023, 48029, 48033, 48034, 48035, 48052, 48053, 48056, 48057, 48058,
    #      48059, 48060, 48061, 48062, 48063, 48064, 48065, 48066, 48067, 48068, 48069, 48070, 48071, 48072, 48073, 48074,
    #      48075],
    #     [48469, 48483, 48484, 48485, 48522, 48523, 48524, 48538, 48539, 48540, 48553, 48564, 48575, 48586, 48597, 48608,
    #      48614, 48626, 48627, 48628, 48665, 48666, 48667, 48681, 48682, 48683, 48698, 48709, 48720, 48731, 48742, 48753,
    #      48759, 48763, 48764, 48765, 48782, 48783, 48786, 48787, 48788, 48789, 48790, 48791, 48792, 48793, 48794, 48795,
    #      48796, 48797, 48798, 48799, 48800, 48801, 48802, 48803, 48804, 48805],
    #     [48323, 48337, 48338, 48339, 48376, 48377, 48378, 48392, 48393, 48394, 48407, 48418, 48429, 48440, 48451, 48462,
    #      48468, 48480, 48481, 48482, 48519, 48520, 48521, 48535, 48536, 48537, 48552, 48563, 48574, 48585, 48596, 48607,
    #      48613, 48617, 48618, 48619, 48636, 48637, 48640, 48641, 48642, 48643, 48644, 48645, 48646, 48647, 48648, 48649,
    #      48650, 48651, 48652, 48653, 48654, 48655, 48656, 48657, 48658, 48659],
    #     [48177, 48191, 48192, 48193, 48230, 48231, 48232, 48246, 48247, 48248, 48261, 48272, 48283, 48294, 48305, 48316,
    #      48322, 48334, 48335, 48336, 48373, 48374, 48375, 48389, 48390, 48391, 48406, 48417, 48428, 48439, 48450, 48461,
    #      48467, 48471, 48472, 48473, 48490, 48491, 48494, 48495, 48496, 48497, 48498, 48499, 48500, 48501, 48502, 48503,
    #      48504, 48505, 48506, 48507, 48508, 48509, 48510, 48511, 48512, 48513],
    #     [48817, 48907, 48921, 48922, 48923, 48960, 48961, 48962, 48976, 48977, 48978, 48991, 49002, 49013, 49024, 49035,
    #      49046, 49052, 49064, 49065, 49066, 49101, 49102, 49103, 49115, 49116, 49117, 49129, 49137, 49145, 49153, 49161,
    #      49169, 49173, 49176, 49177, 49178, 49188, 49189, 49190, 49191, 49192, 49193, 49194, 49195, 49196, 49197, 49198,
    #      49199, 49200, 49201, 49202, 49203, 49204, 49205, 49206, 49207, 49208, 49209],
    #     [47899, 47938, 47940, 48030, 48042, 48043, 48044, 48081, 48082, 48083, 48097, 48098, 48099, 48114, 48125, 48136,
    #      48147, 48158, 48169, 48175, 48179, 48180, 48181, 48198, 48199, 48202, 48203, 48204, 48205, 48206, 48207, 48208,
    #      48209, 48210, 48211, 48212, 48213, 48214, 48215, 48216, 48217, 48218, 48219, 48220, 48221],
    #     [47177, 47203, 47204, 47205, 47243, 47244, 47245, 47260, 47261, 47262, 47279, 47293, 47306, 47319, 47332, 47345,
    #      47356, 47381, 47382, 47383, 47406, 47407, 47408, 47488, 47489, 47490, 47491, 47492, 47493, 47494, 47495, 47496,
    #      47497, 47498, 47499, 47500, 47501, 47502, 47606, 47607, 47608, 47609, 47610, 47611, 47612, 47613, 47614, 47615,
    #      47616, 47617, 47618, 47619, 47620, 47621, 47622, 47623, 47624, 47625],
    #     [47420, 47421, 47422, 47529, 47639, 47640, 47641, 47684, 47685, 47686, 47724, 47751, 47778, 47805, 47832, 47859,
    #      47879, 47887, 47888, 47889, 47906, 47907, 47910, 47911, 47912, 47913, 47914, 47915, 47916, 47917, 47918, 47919,
    #      47920, 47921, 47922, 47923, 47924, 47925, 47926, 47927, 47928, 47929],
    #     [47886, 49353, 49354, 49355, 49378, 49379, 49380, 49381, 49382, 49383, 49385, 49387, 49389, 49391, 49393, 49395,
    #      49396, 49397, 49398, 49399, 49411, 49412, 49413, 49414, 49415, 49416, 49417, 49418, 49419, 49420, 49421, 49422,
    #      49423, 49424, 49425, 49426, 49427, 49428, 49429, 49430, 49431, 49432],
    # ]
    # markers = [
    #     [[48906, 48760], [49051, 48905]],
    #     [[48906, 49174], [49051, 49250]],
    #     [[48906, 49251], [49051, 49307]],
    #     [[48906, 47586], [49051, 47883]],
    #     [[48906, 48176], [49051, 48321]],
    #     [[48906, 47880], [49051, 48029]],
    #     [[48906, 48614], [49051, 48759]],
    #     [[48906, 48468], [49051, 48613]],
    #     [[48906, 48322], [49051, 48467]],
    #     [[48906, 49052], [49051, 49173]],
    #     [[48906, 48030], [49051, 48175]],
    #     [[48906, 47356], [49051, 47489]],
    #     # [[48761, 47529], [48906, 47879]],
    #     # [[48761, 47886], [48906, 49396]],
    #     [[48906, 47529], [49051, 47879]],
    #     [[48906, 47886], [49051, 49396]],
    # ]
    source_nodes = [48761, 48775, 48776, 48777, 48814, 48815, 48816, 48830, 48831, 48832, 48845, 48856, 48867, 48878, 48889, 48900,
         48906, 48918, 48919, 48920, 48957, 48958, 48959, 48973, 48974, 48975, 48990, 49001, 49012, 49023, 49034, 49045,
         49051, 49055, 49056, 49057, 49073, 49074, 49076, 49077, 49078, 49079, 49080, 49081, 49082, 49083, 49084, 49085,
         49086, 49087, 49088, 49089, 49090, 49091, 49092, 49093, 49094, 49095]

    # target_nodes = [
    #     [49153, 49161, 49169, 49173, 49176, 49177, 49178, 49188, 49189, 49190, 49191, 49192, 49193, 49194, 49195, 49196,
    #      49197, 49198, 49199, 49200, 49201, 49202, 49203, 49204, 49205, 49206, 49207, 49208, 49209, 48907, 48921, 48922,
    #      48923, 48960, 48961, 48962, 48976, 48977, 48978, 48991, 49002, 49013, 49024, 49035, 49046, 49052, 49064, 49065,
    #      49066, 49101, 49102, 49103, 49115, 49116, 49117, 49129, 49137, 49145],
    #     [48615, 48629, 48630, 48631, 48668, 48669, 48670, 48684, 48685, 48686, 48699, 48710, 48721, 48732, 48743, 48754,
    #      48760, 48772, 48773, 48774, 48811, 48812, 48813, 48827, 48828, 48829, 48844, 48855, 48866, 48877, 48888, 48899,
    #      48905, 48909, 48910, 48911, 48928, 48929, 48932, 48933, 48934, 48935, 48936, 48937, 48938, 48939, 48940, 48941,
    #      48942, 48943, 48944, 48945, 48946, 48947, 48948, 48949, 48950, 48951],
    #     [49536, 49537, 49538, 49557, 49558, 49559, 49560, 49561, 49562, 49563, 49564, 49565, 49566, 49567, 49568, 49569,
    #      49570, 49571, 49572, 49573, 49574, 49575, 49576, 49577, 49578, 47374, 47398, 47399, 47400, 47477, 47478, 47479,
    #      47566, 47569, 47572, 47575, 47578, 47581, 47584, 47585, 47586, 47587, 47588, 47591, 47595, 47658, 47659, 47660,
    #      47703, 47704, 47705, 47737, 47764, 47791, 47818, 47845, 47872, 47883],
    #     [47177, 47203, 47204, 47205, 47243, 47244, 47245, 47260, 47261, 47262, 47279, 47293, 47306, 47319, 47332, 47345,
    #      47356, 47381, 47382, 47383, 47406, 47407, 47408, 47488, 47489, 47490, 47491, 47492, 47493, 47494, 47495, 47496,
    #      47497, 47498, 47499, 47500, 47501, 47502, 47606, 47607, 47608, 47609, 47610, 47611, 47612, 47613, 47614, 47615,
    #      47616, 47617, 47618, 47619, 47620, 47621, 47622, 47623, 47624, 47625],
    #     [48177, 48191, 48192, 48193, 48230, 48231, 48232, 48246, 48247, 48248, 48261, 48272, 48283, 48294, 48305, 48316,
    #      48322, 48334, 48335, 48336, 48373, 48374, 48375, 48389, 48390, 48391, 48406, 48417, 48428, 48439, 48450, 48461,
    #      48467, 48471, 48472, 48473, 48490, 48491, 48494, 48495, 48496, 48497, 48498, 48499, 48500, 48501, 48502, 48503,
    #      48504, 48505, 48506, 48507, 48508, 48509, 48510, 48511, 48512, 48513],
    #     [48323, 48337, 48338, 48339, 48376, 48377, 48378, 48392, 48393, 48394, 48407, 48418, 48429, 48440, 48451, 48462,
    #      48468, 48480, 48481, 48482, 48519, 48520, 48521, 48535, 48536, 48537, 48552, 48563, 48574, 48585, 48596, 48607,
    #      48613, 48617, 48618, 48619, 48636, 48637, 48640, 48641, 48642, 48643, 48644, 48645, 48646, 48647, 48648, 48649,
    #      48650, 48651, 48652, 48653, 48654, 48655, 48656, 48657, 48658, 48659],
    #     [48469, 48483, 48484, 48485, 48522, 48523, 48524, 48538, 48539, 48540, 48553, 48564, 48575, 48586, 48597, 48608,
    #      48614, 48626, 48627, 48628, 48665, 48666, 48667, 48681, 48682, 48683, 48698, 48709, 48720, 48731, 48742, 48753,
    #      48759, 48763, 48764, 48765, 48782, 48783, 48786, 48787, 48788, 48789, 48790, 48791, 48792, 48793, 48794, 48795,
    #      48796, 48797, 48798, 48799, 48800, 48801, 48802, 48803, 48804, 48805],
    #     [49175, 49185, 49186, 49187, 49215, 49216, 49217, 49223, 49224, 49225, 49251, 49256, 49257, 49284, 49285, 49290,
    #      49291, 49295, 49297, 49299, 49301, 49303, 49305, 49307, 49308, 49309, 49315, 49316, 49317, 49319, 49320, 49331,
    #      49332, 49335, 49336, 49349, 49350, 49352, 49358, 49360, 49361, 49362, 49363, 49364, 49365, 49366, 49367, 49370,
    #      49371, 49372, 49373, 47884, 47885, 49070, 49107, 49121],
    #     [47387, 47424, 47425, 47426, 47532, 47533, 47645, 47690, 47880, 47881, 47900, 47901, 47938, 47939, 47940, 47954,
    #      47955, 47956, 48029, 48030, 48033, 48035, 48043, 48044, 48056, 48058, 48059, 48060, 48061, 48062, 48063, 48064,
    #      48065, 48068, 48069, 48082, 48083, 48098, 48099, 48114, 48125, 48136, 48147, 48158, 48169, 48175, 48179, 48198,
    #      48199, 48202, 48204, 48205, 48216, 48217, 48220, 48221],
    #     [49154, 49162, 49170, 49174, 49182, 49183, 49184, 49212, 49213, 49214, 49220, 49221, 49222, 49228, 49232, 49236,
    #      49240, 49244, 49248, 49250, 49252, 49253, 49261, 49262, 49263, 49264, 49265, 49266, 49267, 49268, 49269, 49270,
    #      49271, 49272, 49273, 49274, 49277, 49278, 49279, 49280, 49281, 49282, 49053, 49067, 49068, 49069, 49104, 49105,
    #      49106, 49118, 49119, 49120, 49130, 49138, 49146],
    #     [48031, 48045, 48046, 48047, 48084, 48085, 48086, 48100, 48101, 48102, 48115, 48126, 48137, 48148, 48159, 48170,
    #      48176, 48188, 48189, 48190, 48227, 48228, 48229, 48243, 48244, 48245, 48260, 48271, 48282, 48293, 48304, 48315,
    #      48321, 48325, 48326, 48344, 48345, 48348, 48349, 48350, 48351, 48352, 48353, 48354, 48355, 48356, 48357, 48358,
    #      48359, 48362, 48363, 48364, 48365, 48366, 48367],
    #     [49354, 49355, 49379, 49380, 49382, 49383, 49396, 49397, 49398, 49399, 49411, 49412, 49413, 49415, 49416, 49417,
    #      49418, 49419, 49420, 49421, 49422, 49423, 49424, 49425, 49426, 49427, 49428, 49431, 49432, 47485, 47486, 47487,
    #      47590, 47666, 47667, 47668, 47711, 47712, 47713, 47886],
    #     [47363, 47384, 47385, 47386, 47421, 47422, 47527, 47528, 47529, 47530, 47531, 47534, 47538, 47640, 47641, 47685,
    #      47686, 47879, 47887, 47888, 47889, 47906, 47907, 47910, 47912, 47913, 47914, 47915, 47916, 47917, 47918, 47919,
    #      47920, 47921, 47922, 47923, 47924, 47925, 47928, 47929]
    # ]

    target_nodes = [[16384, 16385, 16386, 16387, 16388, 16389, 16390, 16391, 16128, 16141, 16269, 16270, 16271, 16152, 16286, 16163,
      16297, 16174, 16308, 16057, 16185, 16319, 16196, 16381, 16071, 16072, 16073, 16202, 16330, 16375, 16383, 16341,
      16214, 16215, 16216, 16347, 16382, 16351, 16352, 16353, 16110, 16111, 16112, 16369, 16254, 16370, 16372, 16373,
      16374, 16255, 16376, 16377, 16378, 16379, 16380, 16253, 16126, 16127],
     [73731, 73732, 73733, 73991, 73992, 73995, 73996, 73997, 73998, 73999, 74000, 74001, 73874, 73747, 73748, 73749,
      73875, 73876, 74002, 74003, 74004, 74005, 74006, 74007, 74008, 74009, 74010, 74011, 73762, 73890, 73891, 73892,
      74013, 74014, 73773, 74012, 73907, 73784, 73918, 73795, 73929, 73806, 73678, 73940, 73817, 73692, 73693, 73694,
      73823, 73951, 73962, 73835, 73836, 73837, 73968, 73972, 73973, 73974],
     [43651, 43779, 43913, 43790, 43665, 43666, 43667, 43796, 43924, 43935, 43808, 43809, 43810, 43941, 43945, 43946,
      43947, 43704, 43705, 43706, 43964, 43965, 43968, 43969, 43970, 43971, 43972, 43973, 43974, 43847, 43976, 43977,
      43720, 43721, 43722, 43848, 43849, 43975, 43978, 43979, 43980, 43981, 43982, 43983, 43986, 43735, 43863, 43864,
      43865, 43987, 43746, 43880, 43757, 43984, 43891, 43985, 43768, 43902],
     [53504, 53761, 53630, 53760, 53764, 53762, 53758, 53631, 53763, 53766, 53767, 53759, 53765, 53517, 53645, 53646,
      53647, 53528, 53662, 53539, 53629, 53673, 53550, 53684, 53561, 53433, 53695, 53572, 53447, 53448, 53449, 53578,
      53706, 53717, 53590, 53591, 53592, 53723, 53727, 53728, 53729, 53486, 53487, 53488, 53745, 53746, 53748, 53749,
      53750, 53751, 53752, 53753, 53754, 53755, 53756, 53757, 53502, 53503],
     [57984, 57985, 57986, 57987, 57988, 57989, 57990, 57863, 57736, 57737, 57738, 57864, 57865, 57994, 57995, 57996,
      57997, 57998, 57999, 58002, 58003, 58000, 58001, 57751, 57879, 57880, 57881, 57762, 57896, 57773, 57907, 57784,
      57991, 57992, 57918, 57993, 57667, 57795, 57929, 57806, 57681, 57682, 57683, 57812, 57940, 57951, 57824, 57825,
      57826, 57957, 57961, 57962, 57963, 57720, 57721, 57722, 57980, 57981],
     # [41472, 41473, 41474, 41605, 41609, 41610, 41611, 41368, 41369, 41370, 41628, 41629, 41632, 41633, 41634, 41635,
     #  41636, 41637, 41638, 41511, 41384, 41385, 41386, 41512, 41513, 41643, 41642, 41645, 41644, 41646, 41647, 41650,
     #  41651, 41648, 41649, 41399, 41527, 41528, 41529, 41410, 41544, 41421, 41555, 41432, 41639, 41566, 41640, 41315,
     #  41443, 41641, 41577, 41454, 41329, 41330, 41331, 41460, 41588, 41599],
     # [8706, 8840, 8717, 8851, 8728, 8862, 8611, 8739, 8873, 8750, 8625, 8626, 8627, 8756, 8884, 8895, 8768, 8769, 8770,
     #  8901, 8944, 8905, 8906, 8907, 8664, 8665, 8666, 8924, 8925, 8928, 8929, 8930, 8931, 8932, 8933, 8934, 8807, 8680,
     #  8681, 8682, 8808, 8809, 8823, 8935, 8936, 8937, 8938, 8939, 8940, 8941, 8942, 8943, 8695, 8824, 8825, 8945, 8946,
     #  8947],
     # [6275, 6403, 6537, 6414, 6289, 6290, 6291, 6420, 6548, 6559, 6432, 6433, 6434, 6565, 6569, 6570, 6571, 6328, 6329,
     #  6330, 6588, 6589, 6592, 6593, 6594, 6595, 6596, 6597, 6598, 6471, 6600, 6601, 6344, 6345, 6346, 6472, 6473, 6599,
     #  6602, 6603, 6604, 6605, 6606, 6607, 6608, 6359, 6487, 6488, 6489, 6610, 6611, 6370, 6609, 6504, 6381, 6515, 6392,
     #  6526],
     # [69891, 70025, 69902, 70119, 70036, 69785, 69913, 70047, 69924, 69799, 69800, 69801, 69930, 70058, 70069, 69942,
     #  69943, 69944, 70075, 70079, 70080, 70081, 69838, 69839, 69840, 70097, 70098, 70100, 70101, 70102, 70103, 70104,
     #  70105, 70106, 70107, 70108, 70109, 69854, 69855, 69856, 69981, 69982, 69983, 70116, 70117, 70118, 70110, 70111,
     #  70112, 70113, 70114, 70115, 69869, 69998, 69997, 69999, 69880, 70014],
     # [22784, 22785, 22786, 22917, 22921, 22922, 22923, 22680, 22681, 22682, 22940, 22941, 22944, 22945, 22946, 22947,
     #  22948, 22949, 22950, 22823, 22696, 22953, 22697, 22698, 22824, 22825, 22951, 22952, 22954, 22955, 22957, 22956,
     #  22958, 22959, 22962, 22711, 22839, 22840, 22841, 22963, 22960, 22722, 22961, 22856, 22733, 22867, 22744, 22878,
     #  22627, 22755, 22889, 22766, 22641, 22642, 22643, 22772, 22900, 22911],
        [51337, 51214, 51348, 51421, 51097, 51359, 51111, 51112, 51113, 51242, 51370, 51381, 51254,
         51255, 51256, 51387, 51391, 51392, 51393, 51150, 51151, 51152, 51409, 51410, 51412, 51413,
         51414, 51415, 51416, 51417, 51418, 51419, 51420, 51293, 51294, 51295, 51168, 51166, 51167,
         51422, 51423, 51424, 51425, 51426, 51427, 51430, 51431, 51429, 51428, 51309, 51310, 51311,
         51326
         ],
        [74116, 74378, 74130, 74131, 74132, 74261, 74389, 74400, 74273, 74274, 74275, 74406, 74410,
         74411, 74412, 74169, 74170, 74171, 74426, 74427, 74429, 74431, 74432, 74430, 74433, 74434,
         74435, 74436, 74437, 74439, 74312, 74185, 74186, 74187, 74313, 74314, 74440, 74441, 74442,
         74443, 74444, 74447, 74448, 74445, 74446, 74328, 74329, 74330, 74345, 74356, 74438, 74367
         ],
        [9344, 9345, 9102, 9103, 9104, 9361, 9362, 9364, 9365, 9366, 9367, 9368, 9369, 9370, 9371, 9372,
         9245, 9246, 9247, 9120, 9118, 9119, 9373, 9374, 9375, 9376, 9377, 9378, 9379, 9380, 9381,
         9382, 9261, 9262, 9263, 9383, 9278, 9289, 9300, 9049, 9311, 9063, 9064, 9065, 9194, 9322,
         9333, 9206, 9207, 9208, 9339, 9343
         ],
        [71812, 71687, 71688, 71689, 71818, 71830, 71831, 71832, 71963, 71967, 71968, 71801, 71969,
         72005, 71726, 71727, 71728, 71986, 71987, 71990, 71991, 71992, 71993, 71994, 71995, 71996,
         71997, 71742, 71743, 71744, 71998, 71999, 71869, 71870, 71871, 72006, 72007, 72000, 72001,
         72002, 72003, 72004, 71757, 71886, 71885, 71887, 72008, 72009, 71902, 71779, 71790, 71673
         ],
        [25472, 25729, 25728, 25730, 25731, 25732, 25734, 25735, 25733, 25485, 25613, 25614, 25615,
         25470, 25725, 25471, 25726, 25630, 25727, 25652, 25401, 25663, 25415, 25416, 25417, 25546,
         25685, 25558, 25559, 25560, 25691, 25695, 25696, 25697, 25454, 25455, 25456, 25713, 25714,
         25716, 25717, 25718, 25719, 25721, 25722, 25723, 25724, 25597, 25598, 25599
         ],
    [48615, 48629, 48630, 48631, 48668, 48669, 48670, 48684, 48685, 48686, 48699, 48710, 48721, 48732, 48743, 48754,
     48760, 48772, 48773, 48774, 48811, 48812, 48813, 48827, 48828, 48829, 48844, 48855, 48866, 48877, 48888, 48899,
     48905, 48909, 48910, 48911, 48928, 48929, 48932, 48933, 48934, 48935, 48936, 48937, 48938, 48939, 48940, 48941,
     48942, 48943, 48944, 48945, 48946, 48947, 48948, 48949, 48950, 48951],
    [49053, 49067, 49068, 49069, 49104, 49105, 49106, 49118, 49119, 49120, 49130, 49138, 49146, 49154, 49281, 49282,
     49162, 49170, 49174, 49182, 49183, 49184, 49212, 49213, 49214, 49220, 49221, 49222, 49228, 49232, 49236, 49240,
     49244, 49248, 49250, 49252, 49253, 49254, 49261, 49262, 49263, 49264, 49265, 49266, 49267, 49268, 49269, 49270,
     49271, 49272, 49273, 49274, 49275, 49276, 49277, 49278, 49279, 49280],
    [49215, 49229, 49233, 49241, 49245, 49249, 49251, 49255, 49256, 49257, 49283, 49284, 49285, 49289, 49290, 49291,
     49295, 49297, 49299, 49301, 49303, 49305, 49307, 49309, 49310, 49311, 49315, 49316, 49317, 49318, 49319, 49320,
     49321, 49322, 49323, 49324, 49325, 49326, 49327, 49328, 49329, 49330, 49331, 49332, 49333, 49334, 49335, 49336],
    [47374, 47398, 47399, 47400, 47477, 47478, 47479, 47566, 47569, 47572, 47575, 47578, 47581, 47584, 47585, 47586,
     47587, 47588, 47591, 47595, 47658, 47659, 47660, 47703, 47704, 47705, 47737, 47764, 47791, 47818, 47845, 47872,
     47883, 49536, 49537, 49538, 49557, 49558, 49559, 49560, 49561, 49562, 49563, 49564, 49565, 49566, 49567, 49568,
     49569, 49570, 49571, 49572, 49573, 49574, 49575, 49576, 49577, 49578],
    [48031, 48045, 48046, 48047, 48084, 48085, 48086, 48100, 48101, 48102, 48115, 48126, 48137, 48148, 48159, 48170,
     48176, 48188, 48189, 48190, 48227, 48228, 48229, 48243, 48244, 48245, 48260, 48271, 48282, 48293, 48304, 48315,
     48321, 48325, 48326, 48327, 48344, 48345, 48348, 48349, 48350, 48351, 48352, 48353, 48354, 48355, 48356, 48357,
     48358, 48359, 48360, 48361, 48362, 48363, 48364, 48365, 48366, 48367],
    [47642, 47687, 47725, 47752, 47779, 47806, 47833, 47880, 47896, 47897, 47898, 47935, 47936, 47937, 47951, 47952,
     47953, 47968, 47979, 47990, 48001, 48012, 48023, 48029, 48033, 48034, 48035, 48052, 48053, 48056, 48057, 48058,
     48059, 48060, 48061, 48062, 48063, 48064, 48065, 48066, 48067, 48068, 48069, 48070, 48071, 48072, 48073, 48074,
     48075],
    [48469, 48483, 48484, 48485, 48522, 48523, 48524, 48538, 48539, 48540, 48553, 48564, 48575, 48586, 48597, 48608,
     48614, 48626, 48627, 48628, 48665, 48666, 48667, 48681, 48682, 48683, 48698, 48709, 48720, 48731, 48742, 48753,
     48759, 48763, 48764, 48765, 48782, 48783, 48786, 48787, 48788, 48789, 48790, 48791, 48792, 48793, 48794, 48795,
     48796, 48797, 48798, 48799, 48800, 48801, 48802, 48803, 48804, 48805],
    [48323, 48337, 48338, 48339, 48376, 48377, 48378, 48392, 48393, 48394, 48407, 48418, 48429, 48440, 48451, 48462,
     48468, 48480, 48481, 48482, 48519, 48520, 48521, 48535, 48536, 48537, 48552, 48563, 48574, 48585, 48596, 48607,
     48613, 48617, 48618, 48619, 48636, 48637, 48640, 48641, 48642, 48643, 48644, 48645, 48646, 48647, 48648, 48649,
     48650, 48651, 48652, 48653, 48654, 48655, 48656, 48657, 48658, 48659],
    [48177, 48191, 48192, 48193, 48230, 48231, 48232, 48246, 48247, 48248, 48261, 48272, 48283, 48294, 48305, 48316,
     48322, 48334, 48335, 48336, 48373, 48374, 48375, 48389, 48390, 48391, 48406, 48417, 48428, 48439, 48450, 48461,
     48467, 48471, 48472, 48473, 48490, 48491, 48494, 48495, 48496, 48497, 48498, 48499, 48500, 48501, 48502, 48503,
     48504, 48505, 48506, 48507, 48508, 48509, 48510, 48511, 48512, 48513],
    [48817, 48907, 48921, 48922, 48923, 48960, 48961, 48962, 48976, 48977, 48978, 48991, 49002, 49013, 49024, 49035,
     49046, 49052, 49064, 49065, 49066, 49101, 49102, 49103, 49115, 49116, 49117, 49129, 49137, 49145, 49153, 49161,
     49169, 49173, 49176, 49177, 49178, 49188, 49189, 49190, 49191, 49192, 49193, 49194, 49195, 49196, 49197, 49198,
     49199, 49200, 49201, 49202, 49203, 49204, 49205, 49206, 49207, 49208, 49209],
    [47899, 47938, 47940, 48030, 48042, 48043, 48044, 48081, 48082, 48083, 48097, 48098, 48099, 48114, 48125, 48136,
     48147, 48158, 48169, 48175, 48179, 48180, 48181, 48198, 48199, 48202, 48203, 48204, 48205, 48206, 48207, 48208,
     48209, 48210, 48211, 48212, 48213, 48214, 48215, 48216, 48217, 48218, 48219, 48220, 48221],
    [47177, 47203, 47204, 47205, 47243, 47244, 47245, 47260, 47261, 47262, 47279, 47293, 47306, 47319, 47332, 47345,
     47356, 47381, 47382, 47383, 47406, 47407, 47408, 47488, 47489, 47490, 47491, 47492, 47493, 47494, 47495, 47496,
     47497, 47498, 47499, 47500, 47501, 47502, 47606, 47607, 47608, 47609, 47610, 47611, 47612, 47613, 47614, 47615,
     47616, 47617, 47618, 47619, 47620, 47621, 47622, 47623, 47624, 47625],
    [47420, 47421, 47422, 47529, 47639, 47640, 47641, 47684, 47685, 47686, 47724, 47751, 47778, 47805, 47832, 47859,
     47879, 47887, 47888, 47889, 47906, 47907, 47910, 47911, 47912, 47913, 47914, 47915, 47916, 47917, 47918, 47919,
     47920, 47921, 47922, 47923, 47924, 47925, 47926, 47927, 47928, 47929],
    [47886, 49353, 49354, 49355, 49378, 49379, 49380, 49381, 49382, 49383, 49385, 49387, 49389, 49391, 49393, 49395,
     49396, 49397, 49398, 49399, 49411, 49412, 49413, 49414, 49415, 49416, 49417, 49418, 49419, 49420, 49421, 49422,
     49423, 49424, 49425, 49426, 49427, 49428, 49429, 49430, 49431, 49432],
    ]

    # markers = [
    #     [[48761, 49153]],
    #     [[48761, 48615]],
    #     [[48761, 49536]],
    #     [[48761, 47177]],
    #     [[48761, 48177]],
    #     [[48761, 48323]],
    #     [[48761, 48469]],
    #     [[48761, 49175]],
    #     [[48761, 47387]],
    #     [[48761, 49154]],
    #     [[48761, 48031]],
    #     [[48761, 49354]],
    #     [[48761, 47363]],
    # ]

    # target_nodes = [target_nodes[i] for i in [3, 11, 12, 13]]
    # markers = [markers[i] for i in [3, 11, 12, 13]]
    # target_nodes = [target_nodes[i] for i in [12, 13]]
    # markers = [markers[i] for i in [12, 13]]

    markers = [
        [[48906, 16202], [49051, 16347]],
        [[48906, 73823], [49051, 73968]],
        [[48906, 43796], [49051, 43941]],
        [[48906, 53578], [49051, 53723]],
        [[48906, 57812], [49051, 57957]],
        # [[48906, 41460], [49051, 41605]],
        # [[48906, 8756], [49051, 8901]],
        # [[48906, 6420], [49051, 6565]],
        # [[48906, 69930], [49051, 70075]],
        # [[48906, 22772], [49051, 22917]],
        [[48906, 51242], [49051, 51387]],
        [[48906, 74261], [49051, 74406]],
        [[48906, 9194], [49051, 9339]],
        [[48906, 71818], [49051, 71963]],
        [[48906, 25546], [49051, 25691]],
        [[48906, 48760], [49051, 48905]],
        [[48906, 49174], [49051, 49250]],
        [[48906, 49251], [49051, 49307]],
        [[48906, 47586], [49051, 47883]],
        [[48906, 48176], [49051, 48321]],
        [[48906, 47880], [49051, 48029]],
        [[48906, 48614], [49051, 48759]],
        [[48906, 48468], [49051, 48613]],
        [[48906, 48322], [49051, 48467]],
        [[48906, 49052], [49051, 49173]],
        [[48906, 48030], [49051, 48175]],
        [[48906, 47356], [49051, 47489]],
        [[48906, 47529], [49051, 47879]],
        [[48906, 47886], [49051, 49396]],
    ]

    # markers = [[[source_nodes[0], nodes[0]]] for nodes in target_nodes]


    # target_nodes = target_nodes[0:10]

    prefix = './data/finan512/'
    G = load_json_graph(prefix + 'graph-with-pos.json')

    # embedings = np.zeros((G.nodes.shape[0]))
    # with open(prefix + 'xnetmf.csv') as f:
    #     spamreader = csv.reader(f, delimiter=' ', quotechar='|')
    #     i = 0
    #     for row in spamreader:
    #         if i > 0:
    #             id = row[0]
    #             embeding = row[1:]
    #
    #         i += 1

    # G = nx.subgraph(G, all_nodes)

    # for id in target_nodes[22]:
    #     G.nodes[id]['y'] += 15
    # for id in target_nodes[21]:
    #     G.nodes[id]['y'] += 5
    # for id in target_nodes[13]:
    #     G.nodes[id]['y'] -= 5
    # for id in target_nodes[23]:
    #     G.nodes[id]['y'] -= 15

    G = Graph(G)
    G.nodes = G.nodes.dot(np.array([[-1, 0], [0, -1]]))
    G = G.to_networkx()
    G = nx.relabel_nodes(G, lambda x: int(x))

    source = nx.Graph(G.subgraph(source_nodes))
    source_G = Graph(source)
    deformed_source_G = Graph(SM_layout(source.copy()))

    target_nodes = [target_nodes[7]]
    for i in range(len(target_nodes)):
        target = nx.Graph(G.subgraph(target_nodes[i]))
        deformed_target, markers = generate(source, SM_layout(source.copy()), target)
        save_json_graph(deformed_target, './data/finan512/result/deformed_target' + str(i) + '.json')

    # target_Gs = []
    # for i in range(len(target_nodes)):
    #     target = nx.Graph(G.subgraph(target_nodes[i]))
    #     target_G = Graph(target)
    #     # ### for finan ####
    #     # if i in [1, 2, 3, 4, 6, 10, 11, 12]:
    #     #     target_G.nodes[:, 1] = np.mean(target_G.nodes[:, 1]) - (target_G.nodes[:, 1] - np.mean(target_G.nodes[:, 1]))
    #     #     target_G = Graph(target_G.to_networkx())
    #     # ### for finan ####
    #     target_Gs.append(target_G)

    # R, t = aligning(source_G, deformed_source_G,
    #                 # np.array([[source_G.id2index[str(id)], source_G.id2index[str(id)]] for id in [49051, 48906]]))
    #                 np.array([[index, index] for index in source_G.index2id]))
    # deformed_source_G.nodes = deformed_source_G.nodes.dot(R.T) + t
    # deformed_source_G.nodes = scale(source_G, deformed_source_G)
    #
    # main(prefix, G, source_G, deformed_source_G, target_Gs, markers)

def compare_for_finan():
    source = load_json_graph('./data/finan512/result/interpolation1.json')
    source_G = Graph(source)
    deformed_source = load_json_graph('./data/finan512/result/interpolation2.json')
    deformed_source_G = Graph(deformed_source)
    for i in [0, 7, 8, 12, 15, 23]:
        target = target = load_json_graph('./data/finan512/result/target' + str(i) + '.json')
        target_G = Graph(target)
        deformed_target_Gs, markers = generate_G(source_G, deformed_source_G, target_G)
        j = 0
        for name in deformed_target_Gs:
            deformed_target = deformed_target_Gs[name]['deformed_target_G'].to_networkx()
            print(name, 'filter rate:', deformed_target_Gs[name]['filtered_markers'].shape[0] / np.min(
                (source_G.nodes.shape[0], target_G.nodes.shape[0])))
            save_json_graph(deformed_target, './data/finan512/result/deformed_target' + str(i) + '_' + name + '.json')
            # save_json_graph(deformed_target, './data/finan512/result/deformed_target' + str(i) + '.json')
            j += 1

def main_for_vis():
    source = load_json_graph('./data/vis/result/interpolation1.json')
    source_G = Graph(source)
    deformed_source = load_json_graph('./data/vis/result/interpolation1.json')
    deformed_source_G = Graph(deformed_source)
    for i in range(6):
        target = target = load_json_graph('./data/vis/result/target' + str(i) + '.json')
        target_G = Graph(target)
        deformed_target_Gs, markers = generate_G(source_G, deformed_source_G, target_G)
        j = 0
        for name in deformed_target_Gs:
            deformed_target = deformed_target_Gs[name]['deformed_target_G'].to_networkx()
            print(name, 'filter rate:', deformed_target_Gs[name]['filtered_markers'].shape[0] / np.min((source_G.nodes.shape[0], target_G.nodes.shape[0])))
            # save_json_graph(deformed_target, './data/vis/result/deformed_target' + str(i) + '_' + name + '.json')
            save_json_graph(deformed_target, './data/vis/result/deformed_target' + str(i) + '.json')
            j += 1

if __name__ == '__main__':
    # main_for_power()
    # main_for_cortex()
    # main_for_mouse()
    # main_for_price()
    main_for_vis()
    # main_for_finan()
    # main_for_finan_compare()
    # main_for_vis_compare()
    # main_for_power_compare()
    # main_for_mouse_compare()
    # compare_for_finan()