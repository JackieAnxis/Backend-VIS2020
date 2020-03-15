

def node_id_map(source_G, target_G):
    m = {
        'source': {},
        'target': {}
    }
    for i, n in enumerate(source_G['nodes']):
        m['source'][int(n['id'])] = i
    for i, n in enumerate(target_G['nodes']):
        m['target'][int(n['id'])] = i
    return m


def get_knn_sim_correspondence(node_maps, source_G, target_G):
    id_map = node_id_map(source_G, target_G)
    correspondence = [[] for i in range(len(target_G['nodes']))]
    pairs = [[id_map['target'][int(x)], id_map['source'][int(y)]]
             for x, y in node_maps.items()]
    for t, s in pairs:
        correspondence[t].append(s)
    return correspondence


if __name__ == "__main__":
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

    id_maps = {
        3:3,
        4:4,
        5:5,
        6:6,
        7:2
    }

    corres = get_knn_sim_correspondence(id_maps, source_G, target_G)
    print(corres)
