# -*- coding: UTF-8
import json
import os
import random
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from MT.deform import non_rigid_registration, aligning
from MT.correspondence import build_correspondence_v1, build_correspondence_v2, build_correspondence_v3, build_correspondence_v4
from MT.Graph import Graph
from MT.optimization import merge
from models.utils import load_json_graph, save_json_graph
from fgm.fgm import fgm
from models.layout import tree_layout, radial_tree_layout

names = ["Ga", "Pm", "Sm", "Smac", "IpfpU", "IpfpS", "Rrwm", "FgmU"] # , "FgmD"]

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
            reg_target_G = non_rigid_registration(intermediate_state, target_G, markers, alpha=1, beta=10, gamma=1000, iter=1000)  # deformation
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

    # R, t = aligning(raw_target_G, target_G, np.array([[index, index] for index in target_G.index2id]))
    # target_G.nodes = target_G.nodes.dot(R.T) + t
    target_G.nodes = (target_G.nodes - t).dot(np.linalg.inv(R).T) ############

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

    # R, t = aligning(source_G, deformed_source_G, np.array([[index, index] for index in source_G.index2id]))
    # deformed_source_G.nodes = deformed_source_G.nodes.dot(R.T) + t

    deformed_targets = [deformed_source_G]
    for i in range(len(target_Gs)):
        target_G = target_Gs[i]
        result = modification_transfer(source_G, target_G, markers[i], intermediate_states, inter_res=True)
        deformed_targets.append(result[0])
        deformed_target = result[0].to_networkx()
        align_target = nx.union(result[1]['alignment'].to_networkx(), source_G.to_networkx())

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
            _inter_deformaed_target = inter_deformaed_target_Gs[k].to_networkx()
            inter_deformaed_target = nx.union(_inter_deformaed_target, intermediate_states[k].to_networkx())
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
    G0, G1 = merge(Graph(G), deformed_targets, iter=1000, alpha=0, beta=1, gamma=1000)
    save_json_graph(G0.to_networkx(), prefix + '/result/new.json')
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
        # [[514, 337], [462, 194], [467, 280]],
        # [[514, 265], [462, 344], [467, 250]],
        # [[514, 41], [462, 17], [467, 34]],
        # # [[514, 399], [462, 373], [481, 610]],
        # # [[514, 11], [462, 269], [481, 71]],
        # [[514, 436], [462, 136], [467, 8]],

        [[514, 337], [462, 194], [467, 280]],
        [[514, 565], [462, 382], [467, 49]],
        [[514, 41], [462, 148], [467, 34]],
        # [[514, 399], [462, 373], [481, 610]],
        # [[514, 11], [462, 269], [481, 71]],
        [[514, 270], [462, 136], [467, 9]],
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
                angle = tuple[1]
                graph = modify(graph, node, inter_node, begin=0,
                       stop=angle + (m) * 20 / 180 * np.pi / 2, travaled=travaled)

        return graph

    # def modify_v1(rawgraph):
    #     T = nx.minimum_spanning_tree(rawgraph)
    #     T = radial_tree_layout(T)
    #     for node in T.nodes:
    #         rawgraph.nodes[node]['x'] = T.nodes[node]['x']
    #         rawgraph.nodes[node]['y'] = T.nodes[node]['y']
    #     return rawgraph

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
        # [[1185, 3296]],
        # [[1185, 3755]],
        # [[1185, 4089]],
        # [[1185, 424]],
        # [[1185, 2401]],
        # [[1185, 3592]],
    ]

    source = nx.Graph(G.subgraph(source_nodes))
    source_G = Graph(source)
    deformed_source_G = Graph(modify(source, inter_node=1185, center=1185, travaled={1185}))
    # deformed_source_G = Graph(modify(source))
    H = source
    target_Gs = []
    for i in range(len(target_nodes)):
        target = nx.Graph(G.subgraph(target_nodes[i]))
        H = nx.union(H, target)
        target_G = Graph(target)
        target_Gs.append(target_G)

    main(prefix, H, source_G, deformed_source_G, target_Gs, markers)

def main_for_vis_compare():
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
        # [3753, 3754, 3755, 3756, 3757, 3764, 3765, 3766, 3767, 3768, 3769, 3770, 3771, 3874, 3875],
        # # [957, 958, 959, 960, 961, 1122, 1123, 1124, 1249, 1250, 1283, 1284, 1285, 1329, 1341, 1342, 1353, 1356],
        # # [466, 467, 468, 469, 531, 532, 533, 547, 548, 549, 550],
        # [4085, 4086, 4087, 4088, 4089, 4108, 4109, 4110, 4111, 4112, 4114, 4115, 4116, 4117, 4118],
        # [422, 423, 424, 425, 426, 719, 720, 721, 722, 723, 780, 781, 782, 783, 784, 785, 821, 822, 823],
        # [2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2399, 2400, 2401, 2402, 2459, 2460, 2639, 2640],
        # # [1801, 1802, 1803, 1804, 1805, 1934, 1935, 1936, 1937, 1938, 1978, 1979, 1980, 1981, 1982, 1983],
        # # [2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2879, 2880, 2881, 2882, 2916, 2917, 2918, 2919,
        # #  2920, 2921, 3211, 3212, 3213, 3214]
        # [3390, 3391, 3392, 3393, 3394, 3395, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3702, 3703, 3704, 3705]
    ]
    markers = [
        [[1167, 3531], [1185, 3296]],
        # [[1167, 3770], [1185, 3755]],
        # # [[1167, 959], [1185, 1250]],
        # # [[1167, 466], [1185, 467]],
        # [[1167, 4088], [1185, 4089]],
        # [[1167, 425], [1185, 424]],
        # [[1167, 2363], [1185, 2401]],
        # # [[1167, 1805], [1185, 1934]],
        # # [[1167, 2918], [1185, 2867]],
        # [[1167, 3394], [1185, 3592]],
    ]

    source = nx.Graph(G.subgraph(source_nodes))
    target = nx.Graph(G.subgraph(target_nodes[0]))
    source_G = Graph(source)
    target_G = Graph(target)
    deformed_source_G = Graph(modify(source, inter_node=1185, center=1185, travaled={1185}))
    H = nx.union(source, target)

    source_node_link_data = json.dumps(json_graph.node_link_data(source_G.to_networkx()))
    target_node_link_data = json.dumps(json_graph.node_link_data(target_G.to_networkx()))
    M = fgm(source_node_link_data, target_node_link_data)
    target_Gs = [target_G]


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

def main_for_mouse_compare():
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
    ]
    markers = [[[source_nodes[i], target_nodes[k][i]] for i in [0, 1, -1]] for k in range(len(target_nodes))]

    G = load_json_graph(prefix + 'graph-with-pos.json')
    source = nx.Graph(G.subgraph(source_nodes))
    target = nx.Graph(G.subgraph(target_nodes[0]))
    source_G = Graph(source)
    target_G = Graph(target)
    deformed_source_G = modify(source_G, source_nodes)

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

    G = main(prefix, G, source_G, deformed_source_G, target_Gs, markers)

def main_for_price():
    def modify(graph):
        id2pos = {"3":{"x":603.1899784256054,"y":621.8612844808738},"112":{"x":356.4605163829269,"y":224.3324684324944},"113":{"x":916.455318727647,"y":220.90882652619246},"114":{"x":564.9949590994031,"y":224.54044889591077},"116":{"x":1195.612292042225,"y":228.0732869394473},"117":{"x":186.34624221898628,"y":225.09470045417606},"118":{"x":2.162575231935051,"y":227.79561664404724},"119":{"x":509.69671268929164,"y":226.78346403098794},"120":{"x":224.6607374895325,"y":224.32525359195105},"121":{"x":856.7573107496587,"y":220.154308484966},"122":{"x":140.90001938807922,"y":224.2806651348709},"123":{"x":1078.9793578833169,"y":222.62146219118296},"124":{"x":98.05544990656597,"y":225.01550221312863},"125":{"x":49.863875058267354,"y":227.38734494693028},"126":{"x":454.08349249433826,"y":228.25431797274945},"127":{"x":268.1137708989056,"y":222.25151288894756},"128":{"x":740.3952651835474,"y":223.2924341114691},"129":{"x":972.2371945861883,"y":220.09119409942372},"130":{"x":406.1311459179843,"y":227.55831067953153},"131":{"x":309.80146606879885,"y":223.68639988119645},"132":{"x":683.6194450059195,"y":223.91754703671774},"133":{"x":801.1533597066212,"y":222.72593506231},"134":{"x":617.6553129524744,"y":225.95272608769187},"362":{"x":915.0070044592583,"y":11.060573309627614},"363":{"x":564.6118547791615,"y":9.324568821493529},"380":{"x":1200.6757281873797,"y":-1.4431619894644427},"381":{"x":1151.6275714986987,"y":-0.002693705474456465},"382":{"x":1254.642570446792,"y":-1.0440150649500595},"383":{"x":0.11603971668654367,"y":15.510933639142479},"384":{"x":-76.36233432473796,"y":17.886163099122598},"385":{"x":75.48321281098274,"y":15.629436054402106},"386":{"x":144.5754835675716,"y":14},"387":{"x":1036.0447287112702,"y":0.5539132981597845},"388":{"x":1104.9388739238584,"y":0.6869155216012075},"389":{"x":267.3739382964066,"y":12.554540316891973},"390":{"x":737.3954054377375,"y":7.007825358782611},"391":{"x":406.7485159869151,"y":10.577055239525293},"618":{"x":1200.1686309128486,"y":-183},"619":{"x":1153.2356229142347,"y":-180.94739642846832},"620":{"x":1258.5623863012906,"y":-182.19348478035784},"621":{"x":-0.3241645594162037,"y":-161.4108787750034},"622":{"x":1038.3241645594162,"y":-182.05755371941737}}
        for id in id2pos:
            graph.nodes[int(id)]['x'] = id2pos[id]['x'] #* 2
            graph.nodes[int(id)]['y'] = id2pos[id]['y'] #* 2
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

    source_nodes = [3,112,113,114,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,362,363,380,381,382,383,384,385,386,387,388,389,390,391,618,619,620,621,622]
    target_nodes = [
        [115,364,366,367,368,369,370,371,372,373,374,375,376,377,378,379,603,613,614,615,616,617,765,872],
        [365,604,605,606,607,608,609,610,611,612,758,759,760,761,762,763,764,870,871],
        [5,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,418,419,420,421,422,423,424,677,678],
        [111,354,355,356,357,358,359,360,361,597,598,599,600,601,602,757],
        [2,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,351,352,353,595,596],
        [145,406,407,408,409,410,411,412,413,414,415,416,417,667,668,669,670,671,672,673,674,675,676,833,834,835,836,980],
        [928,966,967,968,969,970,971,972,973,974,975,976,977,978,979,990,991,992,997],
        [4,136,137,138,139,140,141,142,143,144,399,400,401,402,403,404,405,665,666],
        [12,207,210,211,212,213,214,215,216,217,218,219,220,481,482,483,484,485,486,487,488,489,707,708,709,710,711,846,847,848,995],
        [63,303,304,305,306,307,308,309,310,311,312,313,314,315,561,562,563,564,565,566,567,568,569,570,571,572,751,752,753,868,869],
    ]
    markers = [
        # [[3, 115], [118, 368]],
        # [[3, 365], [118, 604]],
        # [[3, 5], [118, 151]],
        # [[3, 111], [118, 354]],
        # [[3, 2], [118, 96]],
        # [[3, 145], [118, 406]],
        # [[3, 928], [118, 966]],
        # [[3, 4], [118, 136]],
        # [[3, 12], [118, 211]],
        # [[3, 63], [118, 304]],
        [[3, 115]],
        [[3, 365]],
        [[3, 5]],
        [[3, 111]],
        [[3, 2]],
        [[3, 145]],
        [[3, 928]],
        [[3, 4]],
        [[3, 12]],
        [[3, 63]]
    ]

    source = nx.Graph(G.subgraph(source_nodes))
    source_G = Graph(source)
    deformed_source_G = Graph(modify(source.copy()))
    # deformed_source_G = Graph(tree_layout(source))
    print(deformed_source_G.to_networkx().nodes.data())

    target_Gs = []
    for i in range(len(target_nodes)):
        target = nx.Graph(G.subgraph(target_nodes[i]))
        target_G = Graph(target)
        target_Gs.append(target_G)

    main(prefix, G, source_G, deformed_source_G, target_Gs, markers)

if __name__ == '__main__':
    # main_for_power()
    # main_for_cortex()
    # main_for_mouse()
    main_for_price()
    # main_for_vis()
    # main_for_vis_compare()
    # main_for_power_compare()
    # main_for_mouse_compare()