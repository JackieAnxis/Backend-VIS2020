import json
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from MT.main import generate
from fgm.fgm import fgm
from models.layout import layout, SM_layout, nx_spring_layout, MMM_layout, GEM_layout
from models.utils import load_json_graph, save_json_graph
import matlab.engine
import csv
from fgm.readcars import read

def relable_from_zero(g):
    i = 0
    label = {}
    for node in sorted(g.nodes, key=lambda n: int(n)):
        label[node] = i
        i += 1
    return nx.relabel_nodes(g, label)

### CMU house dataset ##$
def build_nx_graph(sources, targets, x=[], y=[]):
    G = nx.Graph()
    for i in range(len(sources)):
        G.add_edge(sources[i]-1, targets[i]-1)
    for i in range(len(x)):
        G.nodes[i]['x'] = x[i]
        G.nodes[i]['y'] = y[i]
    return G

def markers2matrix(markers, n, m):
    mat = np.zeros((n, m))
    for marker in markers:
        mat[marker[0], marker[1]] = 1
    return mat

def matrix2markers(matrix):
    markers_tuple = np.nonzero(matrix)
    return np.array(markers_tuple).T

def load_cmu_house_graph():
    ### graph1 ###
    sources = [2, 1, 4, 1, 2, 9, 2, 3, 1, 5, 8, 4, 5, 10, 6, 3, 11, 1, 5, 6, 10, 13, 2, 7, 9, 11, 4, 5, 8, 12, 6, 9, 10, 11, 16, 17, 1, 6, 12, 6, 11, 14, 15, 19, 4, 9, 10, 13, 7, 8, 3, 14, 15, 21, 6, 12, 14, 20, 24, 7, 9, 17, 22, 23, 8, 18, 23, 4, 18, 22, 23, 26, 27, 3, 5, 5, 6, 7, 10, 11, 11, 12, 12, 12, 13, 13, 13, 14, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 28, 28, 28, 28, 28, 28]
    targets = [3, 5, 5, 6, 7, 10, 11, 11, 12, 12, 12, 13, 13, 13, 14, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 28, 28, 28, 28, 28, 28, 2, 1, 4, 1, 2, 9, 2, 3, 1, 5, 8, 4, 5, 10, 6, 3, 11, 1, 5, 6, 10, 13, 2, 7, 9, 11, 4, 5, 8, 12, 6, 9, 10, 11, 16, 17, 1, 6, 12, 6, 11, 14, 15, 19, 4, 9, 10, 13, 7, 8, 3, 14, 15, 21, 6, 12, 14, 20, 24, 7, 9, 17, 22, 23, 8, 18, 23, 4, 18, 22, 23, 26, 27]
    x = [378.209677419355, 168.016129032258, 154.080645161290, 393.306451612903, 373.564516129032, 325.951612903226, 331.177419354839, 415.370967741936, 314.338709677419, 330.016129032258, 208.080645161290, 451.370967741936, 368.338709677419, 326.532258064516, 207.500000000000, 332.338709677419, 257.435483870968, 403.177419354839, 264.403225806452, 385.177419354839, 228.403225806452, 352.661290322581, 342.790322580645, 208.661290322581, 345.693548387097, 303.306451612903, 395.629032258065, 385.177419354839]
    y = [223.274193548387, 132.112903225806, 257.532258064516, 102.500000000000, 189.596774193548, 264.500000000000, 12.4999999999999, 49.6612903225806, 122.822580645161, 150.112903225806, 227.338709677419, 254.629032258065, 173.338709677419, 315.016129032258, 303.403225806452, 198.887096774194, 96.1129032258064, 94.9516129032257, 234.887096774194, 240.693548387097, 306.306451612903, 132.112903225806, 12.4999999999999, 341.145161290323, 321.983870967742, 83.9193548387096, 76.9516129032257, 84.5000000000000]
    G1 = build_nx_graph(sources, targets, x, y)
    ### graph2 ###
    sources = [2, 4, 1, 3, 3, 6, 4, 1, 7, 7, 9, 10, 2, 5, 10, 2, 3, 2, 3, 8, 14, 1, 2, 5, 7, 16, 17, 1, 3, 6, 16, 18, 7, 10, 13, 17, 1, 10, 13, 16, 2, 3, 14, 17, 18, 19, 4, 5, 7, 9, 11, 17, 4, 5, 12, 21, 1, 6, 21, 13, 16, 17, 18, 20, 22, 2, 6, 8, 12, 15, 21, 25, 26, 5, 5, 6, 6, 8, 8, 9, 10, 10, 11, 11, 11, 12, 12, 13, 14, 14, 15, 15, 15, 15, 16, 17, 17, 17, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28]
    targets = [5, 5, 6, 6, 8, 8, 9, 10, 10, 11, 11, 11, 12, 12, 13, 14, 14, 15, 15, 15, 15, 16, 17, 17, 17, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 2, 4, 1, 3, 3, 6, 4, 1, 7, 7, 9, 10, 2, 5, 10, 2, 3, 2, 3, 8, 14, 1, 2, 5, 7, 16, 17, 1, 3, 6, 16, 18, 7, 10, 13, 17, 1, 10, 13, 16, 2, 3, 14, 17, 18, 19, 4, 5, 7, 9, 11, 17, 4, 5, 12, 21, 1, 6, 21, 13, 16, 17, 18, 20, 22, 2, 6, 8, 12, 15, 21, 25, 26]

    x = [371.241935483871, 318.983870967742, 342.209677419355, 129.112903225806, 244.661290322581, 396.209677419355, 145.951612903226, 390.983870967742, 84.4032258064516, 282.403225806452, 116.919354838710, 306.209677419355, 291.693548387097, 337.564516129032, 355.564516129032, 328.854838709677, 239.435483870968, 316.661290322581, 340.467741935484, 275.435483870968, 377.048387096774, 327.112903225806, 317.822580645161, 166.854838709677, 359.629032258065, 421.177419354839, 290.532258064516, 390.983870967742]
    y = [355.080645161290, 162.887096774194, 236.629032258065, 157.080645161290, 143.145161290323, 180.887096774194, 327.209677419355, 180.887096774194, 261.016129032258, 357.403225806452, 321.403225806452, 136.177419354839, 313.274193548387, 187.854838709677, 185.532258064516, 287.725806451613, 248.822580645161, 259.274193548387, 257.532258064516, 339.403225806452, 78.6935483870967, 311.532258064516, 238.951612903226, 237.209677419355, 75.7903225806451, 144.306451612903, 292.951612903226, 159.403225806452]
    G2 = build_nx_graph(sources, targets, x, y)

    ## ground truth ##
    ground_truth_matching = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    ground_truth_matching = np.array(ground_truth_matching)
    return G1, G2, ground_truth_matching

def compute(G1, G2, ground_truth_matching):
    G1 = nx_spring_layout(G1)
    G2 = nx_spring_layout(G2)

    ### match ###
    n = len(G1.nodes)
    m = len(G2.nodes)
    G1 = nx.relabel_nodes(G1, lambda x: str(x))
    G2 = nx.relabel_nodes(G2, lambda x: str(x))
    G1_node_link_data = json.dumps(json_graph.node_link_data(G1))
    G2_node_link_data = json.dumps(json_graph.node_link_data(G2))
    M = fgm(G1_node_link_data, G2_node_link_data)

    names = ["Ga", "Pm", "Sm", "Smac", "IpfpU", "IpfpS", "Rrwm", "FgmU"]
    results = {}
    generated_markers_matrix = np.ones((n, m))
    for i in range(len(names)):
        name = names[i]
        markers = M[name]
        mat = markers2matrix(markers, n, m)
        generated_markers_matrix *= mat
        results[name] = {
            'mat': mat,
            'precision': 0,
            'recall': 0
        }

    generated_markers_matrix = results['FgmU']
    # deformed, markers = generate(G1, G1, G2, markers=matrix2markers(generated_markers_matrix))
    deformed, markers = generate(G1, G1, G2)
    mat = markers2matrix(markers, n, m)
    results['ours'] = {
        'mat': mat,
        'precision': 0,
        'recall': 0
    }
    # ground_truth_matching = mat

    for name in results:
        mat = results[name]['mat']
        right_markers = mat * ground_truth_matching
        mat = mat[np.nonzero(np.sum(ground_truth_matching, axis=1))[0], :][:, np.nonzero(np.sum(ground_truth_matching, axis=0))[0]]
        precision = np.sum(right_markers) / np.sum(mat)  # right markers / all retrived markers
        recall = np.sum(right_markers) / np.sum(ground_truth_matching)  # right markers / all ground truth markers
        results[name]['precision'] = precision
        results[name]['recall'] = precision
        results[name]['nodecount'] = np.sum(ground_truth_matching)
        print(name, precision, recall, np.sum(ground_truth_matching))

    return results

def compare_cars():
    counts = {
        'Cars': 30,
        'Motorbikes': 20
    }
    indexs = {
        'Cars': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30],
        'Motorbikes': [1, 3, 6, 7, 8, 13, 14, 18, 19]
    }
    methods = ["Ga", "Pm", "Sm", "Smac", "IpfpU", "IpfpS", "Rrwm", "FgmU", "ours"]
    for name in ['Cars']:
        accuracies = {
            'Cars': [methods],
            'Motorbikes': [methods],
        }
        recalls = {
            'Cars': [methods],
            'Motorbikes': [methods],
        }
        for outcount in [0, 4, 8, 12, 16, 20]:
            init_result = {
                "precision": [],
                "recall": []
            }
            # for method in methods:
            #     init_result[method] = []
            for index in indexs[name]:  # range(1, counts[name] + 1):
                print('outcount, index: ', outcount, index)
                r = read(index, vehicle=name, outcount=outcount)
                G1 = r['pair'][0]
                G2 = r['pair'][1]

                i = 0
                label1 = {}
                for node in sorted(G1.nodes, key=lambda n: int(n)):
                    label1[node] = i
                    i += 1
                G1 = nx.relabel_nodes(G1, label1)
                i = 0
                label2 = {}
                for node in sorted(G2.nodes, key=lambda n: int(n)):
                    label2[node] = i
                    i += 1
                G2 = nx.relabel_nodes(G2, label2)
                markers = []
                for marker in r['grdt']:
                    markers.append([label1[marker[0]], label2[marker[1]]])
                ground_truth_matching = markers2matrix(markers, len(G1.nodes), len(G2.nodes))
                results = compute(G1, G2, ground_truth_matching)

                prs = []
                rcs = []
                for method in methods:
                    prs.append(results[method]['precision'])
                    rcs.append(results[method]['recall'])
                init_result['precision'].append(prs)
                init_result['recall'].append(rcs)

            with open('./data/correspondence_test/accuracy_' + name + '_' + str(outcount) + '.csv', mode='w') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(methods)
                for row in init_result['precision']:
                    writer.writerow(row)

            with open('./data/correspondence_test/recall_' + name + '_' + str(outcount) + '.csv', mode='w') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(methods)
                for row in init_result['recall']:
                    writer.writerow(row)

if __name__ == '__main__':
    # G1 = load_json_graph('./data/finan512/result/target8.json')
    # G2 = load_json_graph('./data/finan512/result/interpolation1.json')
    # G1 = relable_from_zero(G1)
    # G2 = relable_from_zero(G2)
    # G1 = layout(G1)
    # G2 = layout(G2)
    # ground_truth_matching = np.eye(len(G1.nodes))

    eng = matlab.engine.start_matlab()
    eng.cd(r'./fgm', nargout=0)
    eng.addPath(nargout=0)
    # for interval in []
    methods = ["Ga", "Pm", "Sm", "Smac", "IpfpU", "IpfpS", "Rrwm", "FgmU", "ours"]
    N = 111
    for interval in [1, 20, 40, 60, 80, 100]:
        init_result = {
            "precision": [],
            "recall": []
        }
        for i in range(0, N):
            print('rm_node_count, interval, i: ', 0, interval, i)
            r = eng.readcmu(i + 1, (i + 1 + interval) % (N + 1) + 1, 0)
            G1EG = np.array(r['eg1'], dtype=np.int32)
            G1PT = np.array(r['pt1'], dtype=np.float64)
            G1 = build_nx_graph(G1EG[0], G1EG[1], G1PT[0], G1PT[1])
            G2EG = np.array(r['eg2'], dtype=np.int32)
            G2PT = np.array(r['pt2'], dtype=np.float64)
            G2 = build_nx_graph(G2EG[0], G2EG[1], G2PT[0], G2PT[1])
            ground_truth_matching = np.array(r['grt'])

            results = compute(G1, G2, ground_truth_matching)

            prs = []
            rcs = []
            for method in methods:
                prs.append(results[method]['precision'])
                rcs.append(results[method]['recall'])
            init_result['precision'].append(prs)
            init_result['recall'].append(rcs)

        with open('./data/correspondence_test/accuracy_cmum' + '_' + str(interval) + '_' + str(0) + '.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(methods)
            for row in init_result['precision']:
                writer.writerow(row)

        with open('./data/correspondence_test/recall_cmum' + '_' + str(interval) + '_' + str(0) + '.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(methods)
            for row in init_result['recall']:
                writer.writerow(row)




    # G1, G2, ground_truth_matching = load_cmu_house_graph()
    # G1 = SM_layout(G1)
    # G2 = SM_layout(G2)


    # print(nx.is_connected(G1), nx.is_connected(G2))
    # save_json_graph(G1, './data/correspondence_test/cmum1.json')
    # save_json_graph(G2, './data/correspondence_test/cmum2.json')
    # save_json_graph(deformed, './data/correspondence_test/cmum3.json')
