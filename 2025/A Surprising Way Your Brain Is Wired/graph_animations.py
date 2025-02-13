from manim import *
import networkx as nx
import numpy as np
from numpy.random import default_rng
from artemutils.colormaps import get_colormap, get_continuous_cmap
from copy import deepcopy
import matplotlib.pyplot as plt
from utils import *
from scene_classes import *
from rendering import *


config.frame_rate = 30
config.pixel_width = 3840 
config.pixel_height = 2160



def SCENE_1():
    G = nx.watts_strogatz_graph(100, k=10, p=0.12, seed=4)
    pos = nx.spring_layout(G, seed=42)
    cluster_1_nodes = [11,12,14,15,16,17,18,19]
    cluster_1_edges = [(u, v) for u, v in G.edges() if u in cluster_1_nodes and v in cluster_1_nodes]
    render_wiggling_graph(G, pos, highlight_edges=[], output_name="SCENE_1.1 base_graph_nodes")
    render_edge_growth(G, pos, edges=G.edges(), output_name="SCENE_1.1 base_graph growing edges", edge_creation_time=2)
    render_wiggling_graph(G, pos, highlight_nodes=[], output_name="SCENE_1.1 base_graph edges")
    render_wiggling_graph(G, pos, highlight_nodes=cluster_1_nodes, highlight_edges=cluster_1_edges, output_name="SCENE_1.2 highlighted_cluster_1")
    render_wiggling_graph(G, pos, **highlight_cluster(41), output_name="SCENE_1.3 highlighted_cluster_2")
    render_wiggling_graph(G, pos, **highlight_cluster(62), output_name="SCENE_1.4 highlighted_cluster_3")
    path1 = nx.shortest_path(G, 57, 45)
    render_path_animation(G, pos, path1, output_name='SCENE_1.5 path_1', edge_width=3)
    render_wiggling_graph(G, pos, highlight_nodes=[], highlight_edges=get_edges_in_path(path1), output_name="SCENE_1.5 path_1_highlighted", edge_width=3)

    path2 = nx.shortest_path(G, 14, 65)
    render_path_animation(G, pos, path2, output_name='SCENE_1.6 path_2', edge_width=3)
    render_wiggling_graph(G, pos, highlight_nodes=[], highlight_edges=get_edges_in_path(path2), output_name="SCENE_1.6 path_2_highlighted", edge_width=3)

    path3 = nx.shortest_path(G, 18, 39)
    render_path_animation(G, pos, path3, output_name='SCENE_1.7 path_3', edge_width=3)
    render_wiggling_graph(G, pos, highlight_nodes=[], highlight_edges=get_edges_in_path(path3), output_name="SCENE_1.7 path_3_highlighted", edge_width=3)

    # Create edges between all nodes (fully connected graph)
    G_complete = nx.complete_graph(100)
    render_edge_growth(G_complete, pos, edges=G_complete.edges(),  output_name="SCENE_1.8 edge_growth", edge_creation_time=2, edge_color=GRAY, edge_width=0.25)
    render_wiggling_graph(G_complete, pos, highlight_nodes=[], output_name="SCENE_1.8 edge_growth_wiggle", edge_width=0.25, edge_color=GRAY)


def SCENE_2():
    G = nx.watts_strogatz_graph(100, k=10, p=0.12, seed=4)
    pos = nx.spring_layout(G, seed=42)

    cmap = get_colormap()
    N_paths = 25 # Number of paths to display
    # Generate random paths
    rng = default_rng(43)
    paths = [nx.shortest_path(G, rng.choice(G.nodes()), rng.choice(G.nodes())) for _ in range(N_paths)]
    props = np.linspace(0, 1, N_paths)
    rng.shuffle(props)
    
    render_wiggling_graph(G, pos,  output_name="SCENE_2.1 nodes_and_edges_background", edge_width=0.5, edge_color=GRAY)
    render_multiple_paths(
        G, pos,
        paths=paths,
        path_colors=[ManimColor(cmap(i)) for i in props],
        output_name="SCENE_2.1 multiple_paths",
        edge_width=3,
        node_fade_time=0.2,
        edge_creation_time=0.3,
        path_fade_time=0.3,
        path_display_time=0.4
    )
    render_clustering_analysis(G, pos, [0, 35, 70, 16, 45,1,80,90,31], output_name="SCENE_2.2 clustering_analysis")


def SCENE_3_and_4():
    kwargs = {
        "node_radius": 0.1,
        "edge_width": 1,
        'edge_color': GRAY,
    }
    G = nx.triangular_lattice_graph(10,16)
    pos = nx.get_node_attributes(G, 'pos')
    for key in pos:
        pos[key] = np.array(pos[key])*0.15
    render_wiggling_graph(G, pos, highlight_edges=[], output_name="SCENE_3.0 lattice_graph_nodes", **kwargs)
    render_edge_growth(G, pos, edges=G.edges(), output_name="SCENE_3.1 lattice_graph_edge_growth", **kwargs, edge_creation_time=5)
    render_wiggling_graph(G, pos, output_name="SCENE_3.2 lattice_graph_base", **kwargs)
    idx = default_rng(42).choice(len(G.nodes()), 6, replace=False)
    render_clustering_analysis(G, pos, [list(G.nodes())[i] for i in idx], output_name="SCENE_3.3 lattice_graph_clustering", **kwargs, center_node_fade_time=0.4, neighbor_node_fade_time=0.4, edge_creation_time=0.4, center_edge_width=3, neighbor_edge_width=4, display_time=0.4, fade_time=0.4)

    # Randomly select paths
    paths,colors = select_random_paths(G, 10, threshold=5, seed=42)
    render_multiple_paths(
        G, pos,
        paths=paths,
        path_colors=colors,
        output_name="SCENE_3.4 lattice_graph_multiple_paths",
        edge_width=3.5,
        node_fade_time=0.1,
        edge_creation_time=0.15,
        path_fade_time=0.2,
        path_display_time=0.2
    )


    #  --- Random Graph
    G_lattice = nx.triangular_lattice_graph(10,16)
    n_edges = G_lattice.number_of_edges()
    n_nodes = G_lattice.number_of_nodes()
    pos_lattice = nx.get_node_attributes(G_lattice, 'pos')
    pos = {}
    for k,node in enumerate(pos_lattice.keys()):
        pos[k] = np.array(pos_lattice[node])*0.15 + default_rng(42).uniform(-0.1,0.1,2)

    G = nx.watts_strogatz_graph(n_nodes, k=6, p=1, seed=42)
    render_edge_growth(G, pos, G.edges(), n_groups=5, output_name="SCENE_4.0 random_graph_lagged_edge_growth", edge_creation_time=1, **kwargs)
    render_wiggling_graph(G, pos, highlight_edges=[], output_name="SCENE_4.0 random_graph_nodes", **kwargs)
    render_wiggling_graph(G, pos, output_name="SCENE_4.0 base_graph", **kwargs)
    
    paths, colors = select_random_paths(G, 10, threshold=2, seed=42)
    render_multiple_paths(
        G, pos,
        paths=paths,
        path_colors=colors,
        output_name="SCENE_4.1 random_graph_multiple_paths",
        edge_width=3.5,
        node_fade_time=0.1,
        edge_creation_time=0.15,
        path_fade_time=0.2,
        path_display_time=0.2
    )

    idx = default_rng(42).choice(len(G.nodes()), 10, replace=False)
    render_clustering_analysis(G, pos, [list(G.nodes())[i] for i in idx], output_name="SCENE_4.2 random_graph_clustering", **kwargs,
    center_node_fade_time=0.4, neighbor_node_fade_time=0.4, edge_creation_time=0.4, center_edge_width=3, neighbor_edge_width=4, display_time=0.4, fade_time=0.4)


def SCENE_5():
    G = nx.powerlaw_cluster_graph(100, 3, p=0.96, seed=55)
    pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(len(G.nodes())), iterations=400)
    rng = default_rng(42)
    center_nodes = rng.choice(G.nodes(), 9, replace=False)
    render_wiggling_graph(G, pos, output_name="SCENE_5.0 base_graph", edge_width=1, edge_color=GRAY)
    render_multiple_clusters(G, pos, center_nodes, output_name="SCENE_5.0 multiple_clusters", cluster_fade_time=0.5, cluster_display_time=1, edge_width=1.75,
                             cmap=get_continuous_cmap(['#17ffae', '#2b87ff']))

    paths, colors = select_random_paths(G, 10, threshold=2, seed=42, cmap = get_continuous_cmap(['#ff2b67', '#ff4b1f']))
    render_multiple_paths(G, pos, paths, colors, output_name="SCENE_5.1 multiple_paths", edge_width=1.75, node_fade_time=0.1, edge_creation_time=0.2, path_fade_time=0.3, path_display_time=0.4)

def SCENE_6():
    G = nx.powerlaw_cluster_graph(160, 3, p=1, seed=55)
    pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(len(G.nodes())), iterations=400)

    scale_kwards = {
        "scale_nodes_by_degree": True,
        "min_radius": 0.03,
        "max_radius": 0.15,
    }
    render_wiggling_graph(G, pos, output_name="SCENE_6.0 base_graph", edge_width=1, edge_color=GRAY)
    render_scaling_nodes(G, pos, initial_radius=0.05, min_radius=0.03, max_radius=0.15, duration=4, output_name="SCENE_6.1 scaling_nodes", edge_width=1, edge_color=GRAY)
    render_wiggling_graph(G, pos, output_name="SCENE_6.2 scaled_graph ", edge_width=1, edge_color=GRAY, **scale_kwards)



   # Selecting hubs (4 nodes with highest degree) and edges connected to them
    degrees = dict(G.degree())
    hub_nodes = sorted(degrees, key=degrees.get, reverse=True)[:4]
    for node in hub_nodes:
        hub_edges = [(u, v) for u, v in G.edges() if u == node or v == node]
        render_wiggling_graph(G, pos, highlight_nodes=[node], highlight_edges=hub_edges, output_name="SCENE_6.3 hub {}".format(node), edge_width=1.75, edge_color=RED, node_color=RED, **scale_kwards)


    # Animating node degrees
    selected_nodes = [18, 32, 1, 110, 34, 10, 120, 140, 28, 13, 130, 140, 150,2, 159]
    render_node_degree_animation(G, pos, selected_nodes, output_name="SCENE_6.4 node_degree", cmap=get_continuous_cmap(['#17ffae', '#2b87ff']), edge_width=1.75, **scale_kwards)

    # Animating clusters
    rng = default_rng(42)
    center_nodes = rng.choice(G.nodes(), 9, replace=False)
    render_multiple_clusters(G, pos, center_nodes, output_name="SCENE_6.5 multiple_clusters", cluster_fade_time=0.4, cluster_display_time=0.75, edge_width=1.75, cmap=get_continuous_cmap(['#17ffae', '#2b87ff']), **scale_kwards)

    # Animating paths
    paths, colors = select_random_paths(G, 10, threshold=2, seed=42, cmap=get_continuous_cmap(['#ff2b67', '#ff4b1f']))
    render_multiple_paths(G, pos, paths, colors, output_name="SCENE_6.6 multiple_paths", edge_width=1.75, node_fade_time=0.1, edge_creation_time=0.2, path_fade_time=0.3, path_display_time=0.4, **scale_kwards)


    # Longer paths with different colors
    paths, colors = select_random_paths(G, 15, threshold=4, seed=42, cmap=get_colormap())
    render_multiple_paths(G, pos, paths, colors, output_name="SCENE_6.7 multiple_paths_long", edge_width=1.75, node_fade_time=0.1, edge_creation_time=0.2, path_fade_time=0.3, path_display_time=0.4, **scale_kwards)

    # Edge growth
    render_edge_growth(G, pos, G.edges(), output_name="SCENE_6.8 edge_growth", edge_creation_time=2.5, **scale_kwards, edge_color=GRAY, edge_width=1.75)
    render_wiggling_graph(G, pos, highlight_edges=[], output_name="SCENE_6.8 nodes only", **scale_kwards)


    # Node removal
    rng = default_rng(43)
    nodes_to_remove = []
    while len(nodes_to_remove) < 10:
        node = rng.choice(list(G.nodes()))
        if node not in nodes_to_remove and G.degree(node) < 10:
            nodes_to_remove.append(node)
    render_node_failure(G, pos, nodes_to_remove, output_name="SCENE_6.9 node_failure", **scale_kwards,  edge_color="#363636", node_color= "#878787", display_time=0.75, initial_wait=0, between_wait=0)



def SCENE_7():
    style = {
        "edge_width": 2.5,
        "edge_color": GRAY,
        "node_radius": 0.1
    }
    # Lattice graph
    G_left, pos_left = convert_triangular_lattice(10, 16, scale=0.15)
    render_wiggling_graph(G_left, pos_left, output_name="SCENE_7.0 lattice_base", **style)
    center_nodes = default_rng(42).choice(G_left.nodes(), 10, replace=False)
    render_multiple_clusters(G_left, pos_left, center_nodes, output_name="SCENE_7.1 lattice_clusters", cluster_fade_time=0.4, cluster_display_time=0.75, cmap=get_continuous_cmap(['#17ffae', '#2b87ff']), **style)

    # Small-world graph with hubs
    G_mid = nx.powerlaw_cluster_graph(160, 3, p=1, seed=55)
    pos_mid = nx.spring_layout(G_mid, seed=42, k=1/np.sqrt(len(G_mid.nodes())), iterations=400)
    render_wiggling_graph(G_mid, pos_mid, output_name="SCENE_7.2 small_world_base", edge_width=2.5, edge_color=GRAY, scale_nodes_by_degree=True, min_radius=0.07, max_radius=0.15)
    center_nodes = default_rng(42).choice(G_mid.nodes(), 10, replace=False)
    render_multiple_clusters(G_mid, pos_mid, center_nodes, output_name="SCENE_7.3 small_world_clusters", edge_width=2.5, cluster_fade_time=0.4, cluster_display_time=0.75, cmap=get_continuous_cmap(['#17ffae', '#2b87ff']), scale_nodes_by_degree=True, min_radius=0.07, max_radius=0.15)

    # Random graph
    G_right = nx.erdos_renyi_graph(160, 0.05, seed=42)
    pos_right = nx.spring_layout(G_right, seed=42, k=1/np.sqrt(len(G_right.nodes())), iterations=400)
    render_wiggling_graph(G_right, pos_right, output_name="SCENE_7.4 random_base", **style)
    center_nodes = default_rng(42).choice(G_right.nodes(), 10, replace=False)
    render_multiple_clusters(G_right, pos_right, center_nodes, output_name="SCENE_7.5 random_clusters", cluster_fade_time=0.4, cluster_display_time=0.75, cmap=get_continuous_cmap(['#17ffae', '#2b87ff']), **style)


def SCENE_8():
    G = nx.powerlaw_cluster_graph(250, 3, p=1, seed=55)
    pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(len(G.nodes())), iterations=400)

    scale_kwards = {
        "scale_nodes_by_degree": True,
        "min_radius": 0.03,
        "max_radius": 0.1,
    }

    style_kwargs = {
        "edge_width": 1.25,
        "edge_color": "#363636",
        "node_color" : "#878787"
    }
    render_wiggling_graph(G, pos, output_name="SCENE_8.0 base_graph", **style_kwargs, **scale_kwards)

    # Randomly select nodes to remove
    rng = default_rng(43)
    nodes_to_remove = []
    while len(nodes_to_remove) < 10:
        node = rng.choice(list(G.nodes()))
        if node not in nodes_to_remove and G.degree(node) < 10:
            nodes_to_remove.append(node)
    render_node_failure(G, pos, nodes_to_remove, output_name="SCENE_8.1 node_failure", **scale_kwards, **style_kwargs)


    G_removed = deepcopy(G)
    G_removed.remove_nodes_from(nodes_to_remove)
    render_wiggling_graph(G, pos, highlight_nodes=list(G_removed.nodes()), highlight_edges=list(G_removed.edges()),
                          output_name="SCENE_8.2 base_graph_removed", **style_kwargs, **scale_kwards)


    # Highlighting a cluster
    center_idx = 67
    nodes, edges = get_local_cluster(G_removed, center_idx)
    render_wiggling_graph(G, pos, highlight_nodes=nodes, highlight_edges=edges, output_name="SCENE_8.3 cluster_highlight", **scale_kwards, node_color="#26ffa8", edge_color="#26ffa8", edge_width=2.5)

    # Remove hub nodes
    degrees = dict(G_removed.degree())
    hub_nodes = sorted(degrees, key=degrees.get, reverse=True)[:8]

    render_wiggling_graph(G_removed, pos, output_name="SCENE_8.4 base_graph_removed", **scale_kwards, **style_kwargs)
    render_node_failure(G, pos, hub_nodes,
                        output_name="SCENE_8.4 hub_nodes_failure", **scale_kwards, **style_kwargs)




if __name__ == "__main__":

    # Comment or uncomment lines depending on which scenes you want to render

    # --- SCENE 1 ---
    SCENE_1()

    # --- SCENE 2 ---
    SCENE_2()

    # --- SCENE 3 + 4 --- Lattice vs Random graph
    SCENE_3_and_4()

    # --- SCENE 5 --- Multiple clusters (overview of small-world networks)
    SCENE_5()

    # --- SCENE 6 --- Hubs
    SCENE_6()

    # --- SCENE 7 --- Comparison of different network types
    SCENE_7()

    # --- SCENE 8 --- Robustness to failure
    SCENE_8()
