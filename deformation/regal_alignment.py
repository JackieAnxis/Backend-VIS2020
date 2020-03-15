import pandas as pd
import numpy as np


def node_id_map(source_G, target_G):
    m = {
        'source': {},
        'source_inv': {},
        'target': {},
        'target_inv': {}
    }
    for i, n in enumerate(source_G['nodes']):
        m['source'][int(n['id'])] = i
        m['source_inv'][i] = int(n['id'])
    for i, n in enumerate(target_G['nodes']):
        m['target'][int(n['id'])] = i
        m['target_inv'][i] = int(n['id'])
    return m


def get_embedding(path):
    embed = np.genfromtxt(path, delimiter=',', skip_header=True)
    return embed[:, 1:]


def get_regal_correspondence(source_G, target_G):
    len_source = len(source_G['nodes'])
    len_target = len(target_G['nodes'])
    id_map = node_id_map(source_G, target_G)

    # TODO: fixed embedding path, for test
    embedding_path = './data/bn-mouse-kasthuri/xnetmf.csv'
    embed = get_embedding(embedding_path)

    embed_source = np.array([embed[id_map['source_inv'][i]]
                             for i in range(len_source)])
    embed_target = np.array([embed[id_map['target_inv'][i]]
                             for i in range(len_target)])

    sim_mat = embed_target @ embed_source.T

    corres = np.vstack(np.argmax(sim_mat, axis=1))

    corres_id_pair = []

    for i, r in enumerate(corres):
        for c in r:
            corres_id_pair.append(
                [id_map['target_inv'][i], id_map['source_inv'][c]])

    return corres, corres_id_pair


if __name__ == "__main__":
    # TODO: fixed embedding path, for test
    embedding_path = './data/bn-mouse-kasthuri/xnetmf.csv'
    embed = get_embedding(embedding_path)
    print(embed.shape)
    # print(embed[:5, :5])

    source_G = {
        'nodes': [
            {'id': 2},
            {'id': 3},
            {'id': 4},
            {'id': 5},
            {'id': 6},
        ]
    }

    target_G = {
        'nodes': [
            {'id': 3},
            {'id': 4},
            {'id': 5},
            {'id': 6},
            {'id': 7},
        ]
    }

    corres, id_pairs = get_regal_correspondence(source_G, target_G)
    print(corres)
    print(id_pairs)
