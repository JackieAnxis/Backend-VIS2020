from tulip import tlp
import json
from networkx.readwrite import json_graph
import networkx as nx

def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def load_json_graph(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)

def save_json_graph(G, filename):
    js_graph = json_graph.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(js_graph, f)

def nx2tlp(G):
    graph = tlp.newGraph()
    nodes_map = {}
    for node in G.nodes:
        nodes_map[node] = graph.addNode()
    for edge in G.edges:
        graph.addEdge(nodes_map[edge[0]], nodes_map[edge[1]])
    return graph, nodes_map

def overlap_removal(graph, layout):
    # get a dictionnary filled with the default plugin parameters values
    # graph is an instance of the tlp.Graph class
    params = tlp.getDefaultPluginParameters('Fast Overlap Removal', graph)

    # set any input parameter value if needed
    # params['overlap removal type'] = ...
    params['layout'] = layout
    # params['bounding box'] = ...
    # params['rotation'] = ...
    # params['number of passes'] = ...
    # params['x border'] = ...
    # params['y border'] = ...

    # either create or get a layout property from the graph to store the result of the algorithm
    resultLayout = graph.getLayoutProperty('resultLayout')
    success = graph.applyLayoutAlgorithm('Fast Overlap Removal', resultLayout, params)

    return resultLayout

def SM(tlpgraph):
    params = tlp.getDefaultPluginParameters('Stress Majorization (OGDF)', tlpgraph)
    resultLayout = tlpgraph.getLayoutProperty('resultLayout')
    success = tlpgraph.applyLayoutAlgorithm('FM^3 (OGDF)', resultLayout, params)

    # or store the result of the algorithm in the default Tulip layout property named 'viewLayout'
    # success = graph.applyLayoutAlgorithm('FM^3 (OGDF)', params)
    return resultLayout

def tree(tlpgraph):
    params = tlp.getDefaultPluginParameters('Tree Leaf', tlpgraph)
    # set any input parameter value if needed
    # params['node size'] = ...
    params['orientation'] = 'right to left'
    # params['uniform layer spacing'] = ...
    params['layer spacing'] = 128
    # params['node spacing'] = 9
    resultLayout = tlpgraph.getLayoutProperty('resultLayout')
    success = tlpgraph.applyLayoutAlgorithm('Tree Leaf', resultLayout, params)
    return resultLayout

def radialtree(tlpgraph):
    params = tlp.getDefaultPluginParameters('Tree Radial', tlpgraph)
    # set any input parameter value if needed
    # params['node size'] = ...
    # params['orientation'] = ...
    # params['uniform layer spacing'] = ...
    # params['layer spacing'] = ...
    # params['node spacing'] = ...
    resultLayout = tlpgraph.getLayoutProperty('resultLayout')
    success = tlpgraph.applyLayoutAlgorithm('Tree Radial', resultLayout, params)
    return resultLayout

def MMM(tlpgraph):
    # get a dictionnary filled with the default plugin parameters values
    # graph is an instance of the tlp.Graph class
    # params = tlp.getDefaultPluginParameters('Tree Leaf', tlpgraph)
    params = tlp.getDefaultPluginParameters('MMM Example Nice Layout (OGDF)', tlpgraph)
    # params = tlp.getDefaultPluginParameters('LinLog', tlpgraph)
    # set any input parameter value if needed
    # params['Layer spacing'] = ...
    # params['Node spacing'] = ...

    # either create or get a layout property from the graph to store the result of the algorithm
    resultLayout = tlpgraph.getLayoutProperty('resultLayout')
    # success = tlpgraph.applyLayoutAlgorithm('Tree Leaf', resultLayout, params)
    success = tlpgraph.applyLayoutAlgorithm('MMM Example Nice Layout (OGDF)', resultLayout, params)
    # success = tlpgraph.applyLayoutAlgorithm('LinLog', resultLayout, params)
    return resultLayout

def FM3(tlpgraph):
    # get a dictionnary filled with the default plugin parameters values
    # graph is an instance of the tlp.Graph class
    params = tlp.getDefaultPluginParameters('FM^3 (OGDF)', tlpgraph)

    # set any input parameter value if needed
    # params['Edge Length Property'] = ...
    # params['Node Size'] = ...
    # params['Unit edge length'] = ...
    # params['New initial placement'] = ...
    # params['Fixed iterations'] = ...
    # params['Threshold'] = ...
    # params['Page Format'] = ...
    params['Quality vs Speed'] = 'GorgeousAndEfficient'
    # params['Edge Length Measurement'] = ...
    # params['Allowed Positions'] = ...
    # params['Tip Over'] = ...
    # params['Pre Sort'] = ...
    # params['Galaxy Choice'] = ...
    # params['Max Iter Change'] = ...
    # params['Initial Placement Mult'] = ...
    # params['Force Model'] = ...
    # params['Repulsive Force Method'] = ...
    # params['Initial Placement Forces'] = ...
    # params['Reduced Tree Construction'] = ...
    # params['Smallest Cell Finding'] = ...

    # either create or get a layout property from the graph to store the result of the algorithm
    resultLayout = tlpgraph.getLayoutProperty('resultLayout')
    success = tlpgraph.applyLayoutAlgorithm('FM^3 (OGDF)', resultLayout, params)

    # or store the result of the algorithm in the default Tulip layout property named 'viewLayout'
    # success = graph.applyLayoutAlgorithm('FM^3 (OGDF)', params)
    return resultLayout

def layout(G):
    graph, nodes_map = nx2tlp(G)
    resultLayout = FM3(graph)
    # resultLayout = SM(graph)
    # resultLayout = MMM(graph)
    resultLayout = overlap_removal(graph, resultLayout)
    for n in G.nodes:
        pos = resultLayout[nodes_map[n]]
        G.nodes[n]['x'] = pos[0]
        G.nodes[n]['y'] = pos[1]
    return G

def tree_layout(G):
    graph, nodes_map = nx2tlp(G)
    resultLayout = tree(graph)
    resultLayout = overlap_removal(graph, resultLayout)
    for n in G.nodes:
        pos = resultLayout[nodes_map[n]]
        G.nodes[n]['x'] = pos[0]
        G.nodes[n]['y'] = pos[1]
    return G

def radial_tree_layout(G):
    graph, nodes_map = nx2tlp(G)
    resultLayout = radialtree(graph)
    resultLayout = overlap_removal(graph, resultLayout)
    for n in G.nodes:
        pos = resultLayout[nodes_map[n]]
        G.nodes[n]['x'] = pos[0]
        G.nodes[n]['y'] = pos[1]
    return G

if __name__ == '__main__':
    # path = './bn-mouse-kasthuri/'
    # path = './bn-mouse_visual-cortex_2/'
    # path = './power-662-bus/'
    path = './VIS/'
    filename = path + "graph.edgelist"
    G = nx.read_edgelist(filename, nodetype=int, data=(('weight', float),))

    # path = './email/'
    # filename = path + "graph.json"
    # G = load_json_graph(filename)
    #
    G.to_undirected()
    # remove_subgraphs = filter(lambda c: nx.diameter(c) <= 3, connected_component_subgraphs(G))
    # for rm_sub in list(remove_subgraphs):
    #     G.remove_nodes_from(rm_sub.nodes)
    #
    # G = nx.dorogovtsev_goltsev_mendes_graph(4)

    G = layout(G)
    save_json_graph(G, path+'graph-with-pos.json')
    # graph1 = nx.subgraph(G, [1753,1754,1755,1756,1757,1766,1767,1768,1769,1770,1771,1772,1775,1776,1777,1788,1789,1790,1806,1807,1808,1809,1810,1814,1815,1816,1817,1818,1819,1831,1832,1833,1851,1852,1853,1854,1866,1867,1868,1869,1905,1906,1907,1908,1909,1910,1911,1912,1913,1914,1915,1916,1917,1920,1921,1926,1927,1928,1929,1930,1931,1932,1933,1962,1963,1964,1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1984,1985,2042,2047,2048,2049,2050,2051,2052,2053,2054,2055,2056,2057,2058,2059,2060,2064,2065,2066,2067,2074,2075,2076,2077,2078,2079,2080,2081,2082,2083,2084,2085,2086,2087,2088,2089,2090,2091,2092,2097,2098,2099,2107,2108,2109,2110,2111,2119,2120,2121,2122,2123,2124,2125,2126,2127,2128,2129,2130,2131,2132,2133,2134,2135,2136,2137,2138,2139,2140,2141,2142,2143,2144,2145,2146,2147,2148,2149,2150,2151,2152,2153,2154,2155,2161,2162,2163,2164,2165,2166,2167,2168,2169,2170,2180,2181,2185,2186,2187,2188,2189,2190,2199,2200,2201,2202,2203,2209,2210,2211,2212,2213,2214,2230,2231,2232,2233,2234,2235,2236,2244,2245,2246,2247,2248,2249,2250,2251])
    # graph2 = nx.subgraph(G, [2284,2285,2286,2287,2288,2289,2342,2343,2344,2345,2346,2347,2348,2349,2353,2354,2355,2356,2367,2368,2369,2370,2388,2389,2390,2391,2392,2451,2452,2453,2454,2455,2475,2476,2477,2531,2532,2533,2534,2559,2560,2561,2562,2566,2567,2568,2569,2570,2605,2606,2607,2608,2611,2612,2613,2620,2621,2622,2623,2624,2641,2642,2674,2675,2676,2677,2683,2684,2685,2686,2687,2697,2698,2699,2700,2701,2725,2726,2727,2728,2729,2730,2738,2739,2740,2741,2742,2743,2744,2754,2766,2767,2768,2769,2787,2788,2789,2790,2794,2795,2801,2802,2803,2804,2805,2810,2811,2812,2830])

    # graph1 = nx.path_graph(4)
    # graph2 = nx.star_graph(3)
    # ismags = nx.isomorphism.ISMAGS(graph1, graph2)
    # ismags.is_isomorphic()

    # largest_common_subgraph = list(ismags.largest_common_subgraph())
    # print(largest_common_subgraph)
    # answer = [
    #     {1: 0, 0: 1, 2: 2},
    #     {2: 0, 1: 1, 3: 2}
    # ]
    # answer == largest_common_subgraph
    #
    # ismags2 = nx.isomorphism.ISMAGS(graph2, graph1)
    # largest_common_subgraph = list(ismags2.largest_common_subgraph())
    # answer = [
    #     {1: 0, 0: 1, 2: 2},
    #     {1: 0, 0: 1, 3: 2},
    #     {2: 0, 0: 1, 1: 2},
    #     {2: 0, 0: 1, 3: 2},
    #     {3: 0, 0: 1, 1: 2},
    #     {3: 0, 0: 1, 2: 2}
    # ]
    # answer == largest_common_subgraph