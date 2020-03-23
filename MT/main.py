# -*- coding: UTF-8
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import networkx as nx
from MT.deform import non_rigid_registration, aligning
from MT.correspondence import build_correspondence
from MT.Graph import Graph
from models.utils import load_json_graph, save_json_graph


def interpolate(source_G, deformed_source_G, n):
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
    target_G = modification_transfer(source_G, deformed_source_G, target_G, markers, intermediate_states=[source_G, deformed_source_G])
    return target_G.to_networkx()

def modification_transfer(source_G, target_G, markers, intermediate_states=[], inter_res=False):
    # change id2id markers into the index2index markers
    markers = np.array(markers)  # [source, target]
    origin_markers = markers.copy()
    markers[:, 0] = np.array([source_G.id2index[str(id)] for id in markers[:, 0]])
    markers[:, 1] = np.array([target_G.id2index[str(id)] for id in markers[:, 1]])

    # alignment
    target_G = target_G.copy()
    R, t = aligning(source_G, target_G, markers)
    align_target_G = target_G.copy()
    align_target_G.nodes = target_G.nodes = target_G.nodes.dot(R.T) + t


    # deform to the final state through intermediate states
    deformation_target_Gs = []  # every deformations, return to intermediate results
    inter_markers = []  # every matchings, return to intermediate results
    for intermediate_state in intermediate_states:
        # deformation and matching (target 2 source)
        # until no more correspondece are built
        marker_increasing = True
        while marker_increasing:
            reg_target_G = non_rigid_registration(intermediate_state, target_G, markers)  # deformation
            new_markers = build_correspondence(intermediate_state, reg_target_G, markers)  # matching
            inter_markers.append(new_markers.copy())
            target_G = reg_target_G
            if new_markers.shape[0] <= markers.shape[0]:
                marker_increasing = False
            markers = new_markers

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
        
        return deformed_source_G

    prefix = './data/power-662-bus/'
    G = load_json_graph(prefix + 'graph-with-pos.json')
    source_nodes = [462, 575, 589, 588, 477, 476, 466, 574]
    target_nodes = [
        [222, 220, 221, 257, 195, 194, 181, 182, 183, 245, 246],
        [482, 487, 580, 583, 488],
        [135, 136, 11, 10, 12, 271, 273, 289, 290, 137],
        [28, 30, 228, 306, 60, 59, 61, 317, 31],
        [272, 271, 216, 8, 7, 9, 71, 99, 214, 215, 320],
    ]

    markers = [
        [[462, 220], [589, 257], [466, 245], [477, 181]],
        [[462, 583], [589, 488], [466, 580], [477, 482]],
        [[462, 290], [589, 273], [466, 135], [477, 10]],
        [[462, 317], [589, 60], [466, 28], [477, 228]],
        [[462, 272], [589, 215], [466, 7], [477, 71]],
    ]
    
    source = nx.Graph(G.subgraph(source_nodes))
    source_G = Graph(source)
    deformed_source_G = modify(source_G, source_nodes)
    intermediate_states_count = 5
    intermediate_states = interpolate(source_G, deformed_source_G, intermediate_states_count)
    
    for i in range(len(target_nodes)):
        target = nx.Graph(G.subgraph(target_nodes[i]))
        target_G = Graph(target)
        result = modification_transfer(source_G, target_G, markers[i], intermediate_states, inter_res=True)
        deformed_target = result[0].to_networkx()
        align_target = result[1]['alignment'].to_networkx()

        save_json_graph(target, prefix + '/result/target' + str(i) + '.json')
        save_json_graph(align_target, prefix + '/result/aligned_target' + str(i) + '.json')
        save_json_graph(deformed_target, prefix + '/result/deformed_target' + str(i) + '.json')

        inter_deformaed_target_Gs = result[1]['deformations']
        for k in range(len(inter_deformaed_target_Gs)):
            inter_deformaed_target = inter_deformaed_target_Gs[k].to_networkx()
            save_json_graph(inter_deformaed_target, prefix + '/result/deformed_target' + str(i) + str(k) + '.json')

    for k in range(len(intermediate_states)):
        inter_state = intermediate_states[k].to_networkx()
        save_json_graph(inter_state , prefix + '/result/interpolation' + str(k) + '.json')

if __name__ == '__main__':
    main_for_power()