from manim import *
import networkx as nx
import numpy as np
from numpy.random import default_rng
from artemutils.colormaps import get_colormap, get_continuous_cmap
from copy import deepcopy
import matplotlib.pyplot as plt



def get_local_cluster(G, node):
    nodes = list(G.neighbors(node)) + [node]
    edges = [(u, v) for u, v in G.edges() if u in nodes and v in nodes]
    return nodes, edges

def highlight_cluster(node):
    nodes, edges = get_local_cluster(node)
    return {
        "highlight_nodes": nodes,
        "highlight_edges": edges
    }

def get_edges_in_path(path):
    return [(path[i], path[i+1]) for i in range(len(path)-1)]

def select_random_paths(G, N_paths, threshold=5, seed=42, cmap=get_colormap()):
    rng = default_rng(seed)
    nodes = list(G.nodes())
    paths = []
    while len(paths) < N_paths:
        start, stop = rng.choice(len(nodes)), rng.choice(len(nodes))
        path = nx.shortest_path(G, nodes[start], nodes[stop])
        if len(path) >= threshold:
            paths.append(path)

    props = np.linspace(0, 1, N_paths)
    rng.shuffle(props)
    colors = [ManimColor(cmap(i)) for i in props]
    return paths, colors


def convert_triangular_lattice(m, n, scale=1):
    G = nx.triangular_lattice_graph(m, n)
    pos = nx.get_node_attributes(G, 'pos')


    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    H = nx.Graph()
    
    # Add nodes and edges with new integer labels
    for edge in G.edges():
        u, v = edge
        H.add_edge(node_mapping[u], node_mapping[v])
    
    # Copy edge attributes if any exist
    for old_edge in G.edges():
        new_edge = (node_mapping[old_edge[0]], node_mapping[old_edge[1]])
        H.edges[new_edge].update(G.edges[old_edge])
    
    mapped_pos = {node_mapping[node]: np.array(pos[node])*scale for node in G.nodes()}
    return H, mapped_pos

