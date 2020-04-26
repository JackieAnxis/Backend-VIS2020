import json
import matlab.engine
import numpy as np
import networkx as nx
from MT.Graph import Graph
from MT.deform import aligning
from models.utils import load_json_graph, save_json_graph


def fgm(source, target, name="FgmU"):
    '''

    :param source:
    :param target:
    :param name:
            "Ga"
            "Pm"
            "Pm"
            "Smac"
            "IpfpU"
            "IpfpS"
            "Rrwm"
            "FgmU"
            "FgmD"
    :return:
    '''
    eng = matlab.engine.start_matlab()
    eng.cd(r'./fgm', nargout=0)
    eng.addPath(nargout=0)
    # r = eng.fgm(source, target, name)
    r = eng.fgmOnly(source, target, name)
    # r = eng.rrwmOnly(source, target, name)
    source_index2id = np.array(r['sourceindex2id'][0], dtype=np.int32)
    target_index2id = np.array(r['targetindex2id'][0], dtype=np.int32)

    res = {}
    for name in r:
        if name != 'sourceindex2id' and name != 'targetindex2id':
            x = np.array(r[name]).nonzero()
            cor = np.vstack((source_index2id[np.ix_(x[0])], target_index2id[np.ix_(x[1])]))
            res[name] = cor.T
    return res

if __name__ == '__main__':
    prefix = './data/VIS/result/'
    source_path = prefix + 'interpolation0.json' # json.load(f)
    target_path = prefix + 'target5.json'


    with open(source_path) as f:
        source = json.dumps(json.load(f))
    with open(target_path) as f:
        target = json.dumps(json.load(f))

    res = fgm(source, target)

    source = load_json_graph(source_path)
    target = load_json_graph(target_path)
    source_G = Graph(source)
    target_G = Graph(target)


    for name in res:
        markers = np.zeros(shape=res[name].shape, dtype=np.int32)
        markers[:, 0] = np.array([source_G.id2index[str(id)] for id in res[name][:, 0]], dtype=np.int32)
        markers[:, 1] = np.array([target_G.id2index[str(id)] for id in res[name][:, 1]], dtype=np.int32)
        target_G_copy = target_G.copy()
        R, t = aligning(source_G, target_G_copy, markers)
        target_G_copy.nodes = target_G.nodes.dot(R.T) + t
        H = nx.union(source, target_G_copy.to_networkx())
        for cor in res[name]:
            sid = str(cor[0])
            tid = str(cor[1])
            H.add_edge(sid, tid)
        save_json_graph(H, prefix + name + '.json')