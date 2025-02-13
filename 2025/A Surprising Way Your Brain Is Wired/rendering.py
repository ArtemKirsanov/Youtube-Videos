from manim import *
import networkx as nx
import numpy as np
from numpy.random import default_rng
from artemutils.colormaps import get_colormap, get_continuous_cmap
from copy import deepcopy
import matplotlib.pyplot as plt

from scene_classes import *
from utils import *


def render_wiggling_graph(
    G,
    pos,
    highlight_nodes=None,
    highlight_edges=None,
    duration=3,
    output_name="graph_animation",
    **kwargs
):
    """Render an animation of a wiggling graph with optional highlights."""
    config.output_file = output_name
    scene = WigglingGraph(
        graph=G,
        pos=pos,
        highlight_nodes=highlight_nodes,
        highlight_edges=highlight_edges,
        duration=duration,
        **kwargs
    )
    scene.render()

def render_path_animation(G, pos, path, output_name="path_animation.mp4", **kwargs):
    """Render an animation of a path through the graph."""
    config.output_file = output_name
    scene = PathAnimation(graph=G, pos=pos, path=path, **kwargs)
    scene.render()

def render_edge_growth(G, pos, edges, n_groups=1, output_name="edge_growth.mp4", **kwargs):
    """Render an animation of edges growing simultaneously."""
    config.output_file = output_name
    scene = EdgeGrowth(graph=G, pos=pos, edges=edges,n_groups=n_groups, **kwargs)
    scene.render()

def render_multiple_paths(
    G,
    pos,
    paths,
    path_colors=None,
    output_name="multiple_paths.mp4",
    **kwargs
):
    """
    Render an animation of multiple paths appearing and fading sequentially.
    
    Args:
        G: NetworkX graph
        pos: Graph layout positions
        paths: List of paths, where each path is a list of node indices
        path_colors: List of colors for each path (optional)
        output_name: Output filename
        **kwargs: Additional arguments passed to MultiPathAnimation
    """
    config.output_file = output_name
    scene = MultiPathAnimation(
        graph=G,
        pos=pos,
        paths=paths,
        path_colors=path_colors,
        **kwargs
    )
    scene.render()


def render_clustering_analysis(
    G,
    pos,
    nodes_to_analyze,
    output_name="clustering_analysis.mp4",
    **kwargs
):
    """Render an animation of clustering coefficient analysis."""
    config.output_file = output_name
    scene = ClusteringCoefficientAnimation(
        graph=G,
        pos=pos,
        nodes_to_analyze=nodes_to_analyze,
        **kwargs
    )
    scene.render()

def render_multiple_clusters(
    G,
    pos,
    center_nodes,
    output_name="multiple_clusters.mp4",
    edge_width=1,
    **kwargs
):
    """Render an animation of multiple local clusters."""
    config.output_file = output_name
    scene = MultipleCusters(
        graph=G,
        pos=pos,
        center_nodes=center_nodes,
        edge_width=edge_width,
        **kwargs
    )
    scene.render()


def render_node_degree_animation(
        G, pos, center_nodes, output_name="node_degree.mp4", **kwargs
):
    config.output_file = output_name
    scene = NodeDegreeAnimation(
        graph=G,
        pos=pos,
        center_nodes=center_nodes,
        **kwargs
    )
    scene.render()



def render_scaling_nodes(
        G, pos, initial_radius, min_radius, max_radius, duration=4, output_name="scaling_nodes.mp4", **kwargs
):
    """Render an animation of nodes scaling according to their degree."""
    config.output_file = output_name
    scene = ScalingNodes(
        graph=G,
        pos=pos,
        initial_radius=initial_radius,
        min_radius=min_radius,
        max_radius=max_radius,
        duration=duration,
        **kwargs
    )
    scene.render()


def render_node_failure(
        G, pos, failure_nodes, fade_time=0.5, output_name="node_failure.mp4", failure_color=RED, **kwargs
):
    """Render an animation of node failures."""
    config.output_file = output_name
    scene = NodeFailure(
        graph=G,
        pos=pos,
        failure_nodes=failure_nodes,
        fade_time=fade_time,
        failure_color=failure_color,
        **kwargs
    )
    scene.render()