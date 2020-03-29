# -*- coding: UTF-8
import sys
import shutil
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import networkx as nx
from MT.deform import non_rigid_registration, aligning
from MT.correspondence import build_correspondence
from MT.Graph import Graph
from MT.optimization import merge
from models.utils import load_json_graph, save_json_graph

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


def generate(source_graph, deformed_source_graph, target_graph, markers):
    source_G = Graph(source_graph)
    deformed_source_G = Graph(deformed_source_graph)
    target_G = Graph(target_graph)
    target_G = modification_transfer(source_G, target_G, markers, intermediate_states=[source_G, deformed_source_G])
    return target_G.to_networkx()

def modification_transfer(source_G, target_G, markers, intermediate_states=[], inter_res=False):
    # change id2id markers into the index2index markers
    markers = np.array(markers)  # [source, target]
    origin_markers = markers.copy()
    markers[:, 0] = np.array([source_G.id2index[str(id)] for id in markers[:, 0]])
    markers[:, 1] = np.array([target_G.id2index[str(id)] for id in markers[:, 1]])

    # alignment
    raw_target_G = target_G.copy()
    R, t = aligning(source_G, target_G, markers)
    align_target_G = target_G.copy()
    align_target_G.nodes = target_G.nodes.dot(R.T) + t


    # deform to the final state through intermediate states
    deformation_target_Gs = []  # every deformations, return to intermediate results
    inter_markers = []  # every matchings, return to intermediate results
    for intermediate_state in intermediate_states:
        # deformation and matching (target 2 source)
        # until no more correspondece are built
        marker_increasing = True
        while marker_increasing:
            reg_target_G = non_rigid_registration(intermediate_state, target_G, markers, alpha=5, beta=1, gamma=1000, iter=1000)  # deformation
            new_markers = build_correspondence(intermediate_state, reg_target_G, markers)  # matching
            inter_markers.append(new_markers.copy())
            target_G = reg_target_G
            if new_markers.shape[0] <= markers.shape[0]:
                marker_increasing = False
            markers = new_markers

        R, t = aligning(raw_target_G, target_G, np.array([[index, index] for index in target_G.index2id]))
        target_G.nodes = target_G.nodes.dot(R.T) + t
        # target_G.nodes = (target_G.nodes - t).dot(np.linalg.inv(R).T)
        deformation_target_G = target_G.copy()
        deformation_target_G.nodes = target_G.nodes
        deformation_target_Gs.append(deformation_target_G)

    if inter_res:
        return target_G, {
            "alignment": align_target_G, # the target graph after alignment
            "deformations": deformation_target_Gs, # for each intermediate state, a deformed target is generated
            "matchings": inter_markers, # markers are built iteratively
        }
    else:
        return target_G

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

    source_nodes = [463, 529, 530, 542, 540, 541, 468, 467, 469, 570, 562, 472, 470, 471, 514, 535, 537, 498, 496, 497, 466, 574, 462, 461]
    # [463, 529, 530, 542, 480, 479, 481, 548, 469, 570, 562, 472, 470, 471, 514, 535, 537, 498, 496, 497, 466, 574, 462, 461]
    target_nodes = [
        # [222, 220, 221, 257, 195, 194, 181, 182, 183, 245, 246],
        [428, 264, 181, 194, 195, 257, 221, 220, 222, 280, 171, 170, 172, 337, 428],
        [265, 328, 288, 49, 32, 344, 424, 425, 565, 564, 419, 250, 417, 382, 347, 427],
        [41, 89, 87, 88, 146, 148, 17, 55, 53, 54, 63, 62, 34, 33, 35, 112, 113],
        # [399, 441, 371, 367, 368, 373, 443, 454, 641, 617, 610, 612, 451, 446]
        [71, 9, 7, 8, 216, 271, 12, 10, 11, 136, 135, 137, 138, 139, 269, 270, 436, 381, 101],
    ]
    markers = [
        [[514, 337], [462, 194], [467, 280]],
        [[514, 265], [462, 344], [467, 250]],
        [[514, 41], [462, 17], [467, 34]],
        # [[514, 399], [462, 373], [481, 610]],
        # [[514, 11], [462, 269], [481, 71]],
        [[514, 436], [462, 136], [467, 8]],
    ]

    G = load_json_graph(prefix + 'graph-with-pos.json')
    # print(nx.shortest_path(G, source=71, target=11))
    # print(nx.shortest_path(G, source=11, target=270))
    # print(nx.shortest_path(G, source=270, target=101))
    source = nx.Graph(G.subgraph(source_nodes))
    source_G = Graph(source)
    deformed_source_G = modify(source_G, source_nodes)

    target_Gs = []
    for i in range(len(target_nodes)):
        target = nx.Graph(G.subgraph(target_nodes[i]))
        target_G = Graph(target)
        target_Gs.append(target_G)

    main(prefix, G, source_G, deformed_source_G, target_Gs, markers)

def main_for_mouse():
    def modify(source_G, source_nodes):
        V = source_G.nodes
        n = V.shape[0]
        center = V[source_G.id2index[str(source_nodes[0])]]
        radius = np.mean(np.sqrt(np.sum((V - center) ** 2, axis=1)))
        interval = 2.0 * np.pi / (n - 1)
        deformed_source_G = source_G.copy()
        i = 0
        for id in source_nodes[1:]:
            x = center[0] + radius * np.sin(interval * i)
            y = center[1] - radius * np.cos(interval * i)
            index = deformed_source_G.id2index[str(id)]
            deformed_source_G.nodes[index] = np.array([x, y])
            i += 1


        return deformed_source_G

    prefix = './data/bn-mouse-kasthuri/'
    source_nodes = [720, 941, 943, 939, 942, 944, 940]
    target_nodes = [
        [676, 853, 850, 851, 854, 848, 852, 849, 855],
        [700, 915, 916, 918, 914, 913, 917],
        [660, 782, 780, 784, 789, 786, 781, 787, 783]
    ]
    G = load_json_graph(prefix + 'graph-with-pos.json')

    source = nx.Graph(G.subgraph(source_nodes))
    source_G = Graph(source)
    deformed_source_G = modify(source_G, source_nodes)

    target_Gs = []
    for i in range(len(target_nodes)):
        target = nx.Graph(G.subgraph(target_nodes[i]))
        target_G = Graph(target)
        target_Gs.append(target_G)

    markers = [[[source_nodes[i], target_nodes[k][i]] for i in [0, 1, -1]] for k in range(len(target_nodes))]
    main(prefix, G, source_G, deformed_source_G, target_Gs, markers)

def main(prefix, G, source_G, deformed_source_G, target_Gs, markers):
    # intermediate_states = interpolate_v2(source_G, deformed_source_G, sequence=[941, 940, 943, 944, 942, 939])
    # intermediate_states_count = 4
    # intermediate_states = interpolate_v1(source_G, deformed_source_G, intermediate_states_count)
    intermediate_states = [source_G, deformed_source_G]
    # intermediate_states = [deformed_source_G]

    shutil.rmtree(prefix + "result")
    os.mkdir(prefix + "result")

    deformed_targets = [deformed_source_G]
    for i in range(len(target_Gs)):
        target_G = target_Gs[i]
        result = modification_transfer(source_G, target_G, markers[i], intermediate_states, inter_res=True)
        deformed_targets.append(result[0])
        deformed_target = result[0].to_networkx()
        align_target = result[1]['alignment'].to_networkx()

        target = target_G.to_networkx()
        for node in target.nodes:
            G.nodes[int(node)]['color'] = ['#436dba']
            target.nodes[node]['color'] = ['#436dba']
            align_target.nodes[node]['color'] = ['#436dba']
            deformed_target.nodes[node]['color'] = ['#436dba']

        save_json_graph(target, prefix + '/result/target' + str(i) + '.json')
        save_json_graph(align_target, prefix + '/result/aligned_target' + str(i) + '.json')
        save_json_graph(deformed_target, prefix + '/result/deformed_target' + str(i) + '.json')

        inter_deformaed_target_Gs = result[1]['deformations']
        for k in range(len(inter_deformaed_target_Gs)):
            inter_deformaed_target = inter_deformaed_target_Gs[k].to_networkx()
            for node in inter_deformaed_target.nodes:
                inter_deformaed_target.nodes[node]['color'] = ['#436dba']
            save_json_graph(inter_deformaed_target, prefix + '/result/deformed_target' + str(i) + str(k) + '.json')

    for k in range(len(intermediate_states)):
        inter_state = intermediate_states[k].to_networkx()
        for node in inter_state.nodes:
            G.nodes[int(node)]['color'] = ['#f06f6b']
            inter_state.nodes[node]['color'] = ['#f06f6b']
        save_json_graph(inter_state, prefix + '/result/interpolation' + str(k) + '.json')

    save_json_graph(G, prefix + '/result/pos.json')
    G0, G1 = merge(Graph(G), deformed_targets, iter=1000, alpha=5, beta=1, gamma=200)
    save_json_graph(G0.to_networkx(), prefix + '/result/new.json')

if __name__ == '__main__':
    main_for_power()
    # main_for_mouse()