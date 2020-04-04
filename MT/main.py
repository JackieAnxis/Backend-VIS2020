# -*- coding: UTF-8
import sys
import shutil
import random
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
            reg_target_G = non_rigid_registration(intermediate_state, target_G, markers, alpha=5®, beta=1, gamma=1000, iter=1000)  # deformation
            new_markers = build_correspondence(intermediate_state, reg_target_G, markers)  # matching
            inter_markers.append(new_markers.copy())
            target_G = reg_target_G
            if new_markers.shape[0] <= markers.shape[0]:
                marker_increasing = False
            markers = new_markers

        _markers = markers.copy()
        _markers[:, 0] = np.array([source_G.index2id[marker] for marker in markers[:, 0]])
        _markers[:, 1] = np.array([target_G.index2id[marker] for marker in markers[:, 1]])
        print(_markers)

        deformation_target_G = target_G.copy()
        deformation_target_G.nodes = target_G.nodes
        deformation_target_Gs.append(deformation_target_G)

    # R, t = aligning(raw_target_G, target_G, np.array([[index, index] for index in target_G.index2id]))
    # target_G.nodes = target_G.nodes.dot(R.T) + t
    # target_G.nodes = (target_G.nodes - t).dot(np.linalg.inv(R).T) ############

    if inter_res:
        return target_G, {
            "alignment": align_target_G, # the target graph after alignment
            "deformations": deformation_target_Gs, # for each intermediate state, a deformed target is generated
            "matchings": inter_markers, # markers are built iteratively
        }
    else:
        return target_G

def main(prefix, G, source_G, deformed_source_G, target_Gs, markers):
    # intermediate_states = interpolate_v2(source_G, deformed_source_G, sequence=[672, 836, 834, 838, 839, 831, 830, 829, 835, 832])
    # intermediate_states_count = 8
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
            save_json_graph(inter_deformaed_target, prefix + '/result/deformed_target' + str(i) + '_' + str(k) + '.json')

    for k in range(len(intermediate_states)):
        inter_state = intermediate_states[k].to_networkx()
        for node in inter_state.nodes:
            G.nodes[int(node)]['color'] = ['#f06f6b']
            inter_state.nodes[node]['color'] = ['#f06f6b']
        save_json_graph(inter_state, prefix + '/result/interpolation' + str(k) + '.json')

    save_json_graph(G, prefix + '/result/pos.json')
    G0, G1 = merge(Graph(G), deformed_targets, iter=1000, alpha=0, beta=100, gamma=0)
    save_json_graph(G0.to_networkx(), prefix + '/result/new.json')
    return G0.to_networkx()

def main_for_cortex():
    def modify(source_G, source_nodes):
        V = source_G.nodes
        n = V.shape[0]
        center = V[source_G.id2index[str(source_nodes[0])]]
        radius = np.mean(np.sqrt(np.sum((V - center) ** 2, axis=1))) / 2
        v = np.mean(V[1:], axis=0) - center
        interval = 0.05 * np.pi / (n - 1)
        init_angle = np.arctan(v[1] / v[0]) - interval * (n - 1) / 2
        deformed_source_G = source_G.copy()
        i = 0
        for id in source_nodes[1:]:
            x = center[0] + radius * np.cos(interval * i + init_angle)
            y = center[1] + radius * np.sin(interval * i + init_angle)
            index = deformed_source_G.id2index[str(id)]
            deformed_source_G.nodes[index] = np.array([x, y])
            i += 1

        return deformed_source_G
        V = source_G.nodes
        n = V.shape[0]
        center = np.mean(V[1:], axis=0)
        radius = np.mean(np.sqrt(np.sum((V - center) ** 2, axis=1))) / 2
        deformed_source_G = source_G.copy()
        i = 0
        for id in source_nodes[1:]:
            x = center[0]#  + radius * np.sin(interval * i)
            y = center[1]#  - radius * np.cos(interval * i)
            index = deformed_source_G.id2index[str(id)]
            deformed_source_G.nodes[index] = np.array([x, y])
            i += 1

        return deformed_source_G

    prefix = './data/bn-mouse_visual-cortex_2/'
    source_nodes = [127, 169, 154, 171, 160, 157, 173, 166, 167, 155, 161, 172, 156, 168, 174, 163, 158, 162, 159, 170, 165, 164]
    target_nodes = [
        [7, 29, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 31, 32, 33, 34, 35, 27],
        [8, 38, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 50],
        [177, 190, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 191, 192, 182],
        [102, 153, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 141, 143, 144, 145, 146, 147, 148, 149, 150, 142, 151, 152, 140],
        [94, 107, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 120,
         121, 122, 123, 124, 125, 126, 118],
        [54, 91, 85, 86, 87, 89, 90, 92, 93, 88],
        [83, 98, 95, 96, 97, 100, 99],
        [53, 68, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 58],
        [0, 3, 5, 4, 2, 6, 1],
        # [7, 29, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 31, 32, 33, 27],
        # [8, 38, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 50],
        # [177, 190, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 191, 192, 182],
        # [102, 142, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 141, 143, 144, 145, 146, 147, 148, 149,
        #  150, 140],
        # [94, 107, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 120,
        #  121, 118],
        # [54, 91, 85, 86, 87, 89, 90, 92, 93, 88],
        # [83, 98, 95, 96, 97, 100, 99],
        # [53, 68, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
        #  58],
        # [0, 3, 5, 4, 2],
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
            x = center[0] + radius * np.sin(interval * i) + radius * (random.random()-0.5) / 5
            y = center[1] - radius * np.cos(interval * i) + radius * (random.random()-0.5) / 5
            index = deformed_source_G.id2index[str(id)]
            deformed_source_G.nodes[index] = np.array([x, y])
            i += 1


        return deformed_source_G

    prefix = './data/bn-mouse-kasthuri/'
    source_nodes = [672, 836, 838, 831, 829, 835, 832, 830, 839, 834] # [93, 596, 592, 593, 590, 599, 591, 600, 595, 597, 588, 598, 587, 601]
    target_nodes = [
        [676, 854, 848, 850, 851, 852, 853, 855, 849], ###
        [700, 915, 913, 915, 916, 917, 918, 914],
        [101, 636, 637, 638, 640, 641, 643, 642], ###
        [93, 596, 592, 593, 590, 599, 591, 600, 595, 597, 588, 598, 587, 601],
        [673, 843, 840, 842, 841],
        [974, 980, 982, 981, 983],
        [644, 739, 742, 741, 740],
        [645, 745, 746, 744, 747],
        [722, 950, 951, 952, 953],
        [74, 496, 500, 504, 505], #######
        [98, 623, 625, 626, 627], #######
        [652, 764, 765, 766, 767], #######
        [716, 936, 937, 935],
        [73, 484, 483, 485, 491, 487], ###
        [721, 948, 946, 947, 949, 945],
        [667, 820, 813, 815, 816, 818, 817], ###
        [728, 956, 957, 959, 960, 955, 958, 961],
        [698, 911, 908, 909, 907, 910], ###
        [720, 942, 939, 940, 943, 944, 941], ###
        [691, 890, 887, 888, 889, 891], ###
        [647, 755, 749, 750, 751, 752, 753, 754, 748], ###
        [730, 964, 963, 967, 965, 962, 966], ###
        [660, 782, 781, 787, 783, 780, 786, 784], ###
        [665, 810, 801, 802, 805, 806, 807, 808, 803], ###
        [80, 530, 532, 533, 536, 531], ###
        [88, 567, 569, 570, 573, 572], ###
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
    G = main(prefix, G, source_G, deformed_source_G, target_Gs, markers)

    mapping = {}
    for node in G.nodes:
        mapping[node] = int(node)
    G = nx.relabel_nodes(G, mapping)

    source_nodes = [673, 843, 840, 842, 841]
    target_nodes = [
        [974, 980, 982, 981, 983],
        [644, 739, 742, 741, 740],
        [645, 745, 746, 744, 747],
        [722, 950, 951, 952, 953],
        [74, 496, 500, 504, 505],  #######
        [98, 623, 625, 626, 627],  #######
        [652, 764, 765, 766, 767],  #######
        [716, 936, 937, 935],
    ]

    source = nx.Graph(G.subgraph(source_nodes))
    source_G = Graph(source)
    deformed_source_G = modify(source_G, source_nodes)

    target_Gs = []
    for i in range(len(target_nodes)):
        target = nx.Graph(G.subgraph(target_nodes[i]))
        target_G = Graph(target)
        target_Gs.append(target_G)

    markers = [[[source_nodes[i], target_nodes[k][i]] for i in [0, 1, -1]] for k in range(len(target_nodes))]
    # G = main(prefix, G, source_G, deformed_source_G, target_Gs, markers)

def main_for_vis():
    def modify(rawgraph, inter_node, center, begin = 0, stop = 2 * np.pi, travaled={}):
        graph = rawgraph.copy()
        neighbors = list(nx.neighbors(graph, inter_node))
        n = len(neighbors)
        angles = []
        x0 = graph.nodes[inter_node]['x']
        y0 = graph.nodes[inter_node]['y']
        v0 = np.array([x0, y0])
        radius = 0
        for node in neighbors:
            if node not in travaled:
                x = graph.nodes[node]['x']
                y = graph.nodes[node]['y']
                v = np.array([x, y])
                r = np.sqrt(np.sum((v - v0)**2))
                radius += r
                cosangle = (v-v0).dot([1, 0]) / np.linalg.norm(v-v0)
                angle = np.arccos(cosangle)
                if y < y0:
                    angle = 2 * np.pi - angle

                angles.append([node, angle])

        radius /= n
        angles = sorted(angles, key=lambda x:x[1])
        i = 0
        interval = (stop - begin) / n

        for tuple in angles:
            node = tuple[0]
            angle = begin + i * interval
            x = radius * np.cos(angle) + x0
            y = radius * np.sin(angle) + y0
            graph.nodes[node]['x'] = x
            graph.nodes[node]['y'] = y
            travaled.add(node)
            i += 1

        for tuple in angles:
            node = tuple[0]
            m = len(list(nx.neighbors(graph, node)))
            if m > 0:
                graph = modify(graph, node, inter_node, begin=angle - (m-2) * 20 / 180 * np.pi / 2,
                       stop=angle + (m-2) * 20 / 180 * np.pi / 2, travaled=travaled)

        return graph


    prefix = './data/VIS/'
    G = load_json_graph(prefix + 'graph-with-pos.json')

    source_nodes = [1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1182, 1183, 1184, 1185, 1256, 1257, 1258, 1388, 1389, 1390, 1391, 1392, 1393, 1394]
    target_nodes = [
        [3294, 3295, 3296, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3653, 3654, 3655, 3656, 3657, 3718, 3719, 3720, 3721, 3722],
        [3753, 3754, 3755, 3756, 3757, 3764, 3765, 3766, 3767, 3768, 3769, 3770, 3771, 3874, 3875],
        # [957, 958, 959, 960, 961, 1122, 1123, 1124, 1249, 1250, 1283, 1284, 1285, 1329, 1341, 1342, 1353, 1356],
        # [466, 467, 468, 469, 531, 532, 533, 547, 548, 549, 550],
        [4085, 4086, 4087, 4088, 4089, 4108, 4109, 4110, 4111, 4112, 4114, 4115, 4116, 4117, 4118],
        [422, 423, 424, 425, 426, 719, 720, 721, 722, 723, 780, 781, 782, 783, 784, 785, 821, 822, 823],
        [2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2399, 2400, 2401, 2402, 2459, 2460, 2639, 2640],
        # [1801, 1802, 1803, 1804, 1805, 1934, 1935, 1936, 1937, 1938, 1978, 1979, 1980, 1981, 1982, 1983],
        # [2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2879, 2880, 2881, 2882, 2916, 2917, 2918, 2919,
        #  2920, 2921, 3211, 3212, 3213, 3214]
        [3390, 3391, 3392, 3393, 3394, 3395, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3702, 3703, 3704, 3705]
    ]
    markers = [
        [[1167, 3531], [1185, 3296]],
        [[1167, 3770], [1185, 3755]],
        # [[1167, 959], [1185, 1250]],
        # [[1167, 466], [1185, 467]],
        [[1167, 4088], [1185, 4089]],
        [[1167, 425], [1185, 424]],
        [[1167, 2363], [1185, 2401]],
        # [[1167, 1805], [1185, 1934]],
        # [[1167, 2918], [1185, 2867]],
        [[1167, 3394], [1185, 3592]],
    ]

    source = nx.Graph(G.subgraph(source_nodes))
    source_G = Graph(source)
    deformed_source_G = Graph(modify(source, inter_node=1185, center=1185, travaled={1185}))
    H = source

    target_Gs = []
    for i in range(len(target_nodes)):
        target = nx.Graph(G.subgraph(target_nodes[i]))
        H = nx.union(H, target)
        target_G = Graph(target)
        target_Gs.append(target_G)

    # markers = [[[source_nodes[i], target_nodes[k][i]] for i in [0, 1, -1]] for k in range(len(target_nodes))]
    main(prefix, H, source_G, deformed_source_G, target_Gs, markers)

if __name__ == '__main__':
    # main_for_power()
    # main_for_cortex()
    # main_for_mouse()
    main_for_vis()