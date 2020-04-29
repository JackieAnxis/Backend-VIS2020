import scipy.io
from scipy.spatial import Delaunay
import random
import numpy as np
import networkx as nx

def read(index, vehicle='Cars', outcount=0):
    name = 'Data_Pairs_' + vehicle
    prefix = './data/Data_for_Cars_and_Motorbikes/' + name + '/'
    path = prefix + 'pair_' + str(index) + '.mat'
    mat = scipy.io.loadmat(path)
    ground_truth = mat['gTruth']
    pts = [
        mat['features1'][:,0:2],
        mat['features2'][:,0:2],
    ]
    tris = [
        Delaunay(pts[0]),
        Delaunay(pts[1]),
    ]
    Gs = []
    for k in range(2):
        nodes = pts[k]
        t = tris[k]
        edges = []
        # m = dict(enumerate(nodes)) # mapping from vertices to nodes
        for i in range(t.nsimplex):
            edges.append([t.vertices[i, 0], t.vertices[i, 1]])
            edges.append([t.vertices[i, 1], t.vertices[i, 2]])
            edges.append([t.vertices[i, 2], t.vertices[i, 0]])
            # edges.append( (m[t.vertices[i,0]], m[t.vertices[i,1]]) )
            # edges.append( (m[t.vertices[i,1]], m[t.vertices[i,2]]) )
            # edges.append( (m[t.vertices[i,2]], m[t.vertices[i,0]]) )

        # build graph
        G = nx.Graph()

        G = nx.Graph(edges)
        for i in range(nodes.shape[0]):
            G.nodes[i]['x'] = nodes[i][0]
            G.nodes[i]['y'] = nodes[i][1]

        # pointIDXY = dict(zip(range(len(nodes.tolist())), nodes.tolist()))
        # nx.draw(G, pointIDXY)
        # plt.show()

        Gs.append(G)


    G1 = Gs[0]
    G2 = Gs[1]
    inliers = [
        list(range(ground_truth.shape[1])),
        (ground_truth[0] - 1).tolist()
    ]
    subG1 = G1.subgraph(inliers[0])
    subG2 = G2.subgraph(inliers[1])
    k = 0
    result = {
        'pair': [],
        'grdt': np.array(inliers).transpose().tolist()
    }
    for subG in [subG1, subG2]:
        potential_outliers = []
        for node in subG.nodes:
            for adj in Gs[k][node]:
                if not subG.has_node(adj):
                    potential_outliers.append(adj)
        potential_outliers
        outliers = random.choices(potential_outliers, k=outcount)
        result['pair'].append(Gs[k].subgraph(inliers[k] + outliers))
        k += 1
    return result

if __name__ == '__main__':
    r = read(1, outcount=5)
    print(r)