
from manim import *
import networkx as nx
import numpy as np
from numpy.random import default_rng
from artemutils.colormaps import get_colormap, get_continuous_cmap
from copy import deepcopy
import matplotlib.pyplot as plt

from utils import *

class BaseGraphScene(Scene):
    """Base class for graph animation scenes with common functionality."""
    
    def __init__(
        self,
        graph,
        pos,
        node_color=WHITE,
        node_radius=0.05,
        edge_color=WHITE,
        edge_width=1,
        scale=4,
        wiggle_amplitude=0.075,
        wiggle_period=3,
        seed=42,
        scale_nodes_by_degree=False,
        min_radius=0.05,
        max_radius=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.G = graph
        self.raw_pos = pos
        self.node_color = node_color
        self.node_radius = node_radius
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.scale = scale
        self.wiggle_amplitude = wiggle_amplitude
        self.wiggle_period = wiggle_period
        self.seed = seed
        self.scale_nodes_by_degree = scale_nodes_by_degree
        self.min_radius = min_radius
        self.max_radius = max_radius

    def setup_graph(self):
        """Initialize graph positions and wiggle parameters."""
        # Convert positions to manim coordinates
        node_list = list(self.raw_pos.keys())
        self.pos_array = np.array([self.raw_pos[node] for node in node_list])
        self.pos_array = (self.pos_array - self.pos_array.mean(axis=0)) * self.scale
        self.pos_array = np.column_stack((self.pos_array, np.zeros(len(node_list))))

        # Setup wiggle parameters
        self.num_nodes = len(node_list)
        rng = default_rng(seed=self.seed)
        self.random_angles = rng.uniform(0, 2*np.pi, self.num_nodes)
        self.direction_vectors = np.column_stack((
            np.cos(self.random_angles),
            np.sin(self.random_angles),
            np.zeros(self.num_nodes)
        ))
        self.phase_offsets = rng.uniform(0, 2*np.pi, self.num_nodes)

    def get_node_position(self, node_id):
        """Calculate current position of a node including wiggle effect."""
        time = self.renderer.time
        i = list(self.raw_pos).index(node_id)
        offset = self.wiggle_amplitude * np.sin(
            2 * PI * time/self.wiggle_period + self.phase_offsets[i]
        )
        return self.pos_array[i] + self.direction_vectors[i] * offset

    def get_node_updater(self, node_id):
        """Create an updater function for node position."""
        def updater(mob, dt):
            mob.move_to(self.get_node_position(node_id))
        return updater

    def get_edge_updater(self, start_node, end_node):
        """Create an updater function for edge position."""
        def updater(mob, dt):
            mob.put_start_and_end_on(
                self.get_node_position(start_node),
                self.get_node_position(end_node)
            )
        return updater

    def create_node(self, node_id, color=None, radius=None):
        """Create a node with position updates."""
        if color is None:
            color = self.node_color
        if radius is None:
            if self.scale_nodes_by_degree:
                degree = self.G.degree[node_id]
                max_degree = max(dict(self.G.degree).values())
                radius = self.min_radius + (self.max_radius - self.min_radius) * (degree / max_degree)
            else:
                radius = self.node_radius

        i = list(self.G.nodes()).index(node_id)
        dot = Dot(
            point=self.pos_array[i],
            radius=radius,
            color=color
        )
        dot.add_updater(self.get_node_updater(node_id))
        return dot

    def create_edge(self, start_node, end_node, color=None, width=None):
        """Create an edge with position updates."""
        if color is None:
            color = self.edge_color
        if width is None:
            width = self.edge_width

        start_idx = list(self.G.nodes()).index(start_node)
        end_idx = list(self.G.nodes()).index(end_node)
        edge = Line(
            start=self.pos_array[start_idx],
            end=self.pos_array[end_idx],
            stroke_width=width,
            color=color
        )
        edge.add_updater(self.get_edge_updater(start_node, end_node))
        return edge

    def wait_for_wiggle_period(self):
        """Wait until the current wiggle period completes."""
        time_to_wait = (self.renderer.time // self.wiggle_period + 1) * self.wiggle_period - self.renderer.time
        self.wait(time_to_wait)





class WigglingGraph(BaseGraphScene):
    """Scene for displaying a wiggling graph with optional highlights."""
    
    def __init__(self, highlight_nodes, highlight_edges, duration=3, **kwargs):
        super().__init__(**kwargs)
        self.highlight_nodes = highlight_nodes
        self.highlight_edges = highlight_edges
        self.duration = duration

    def construct(self):
        self.setup_graph()
        highlight_nodes = getattr(self, 'highlight_nodes', None)
        highlight_edges = getattr(self, 'highlight_edges', None)
        
        nodes_to_display = self.G.nodes() if highlight_nodes is None else highlight_nodes
        edges_to_display = self.G.edges() if highlight_edges is None else highlight_edges

        nodes = [self.create_node(node) for node in nodes_to_display]
        edges = [self.create_edge(u, v) for u, v in edges_to_display]
        
        self.add(*edges)
        self.add(*nodes)
        self.wait(self.duration)


class PathAnimation(BaseGraphScene):
    """Scene for animating a path through the graph."""
    
    def __init__(self, path, edge_creation_time=0.5, node_fade_time=0, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.edge_creation_time = edge_creation_time
        self.node_fade_time = node_fade_time

    def construct(self):
        self.setup_graph()

        if self.node_fade_time > 0:    
            current_node = self.create_node(self.path[0])
            self.play(FadeIn(current_node))
            
        for i in range(len(self.path)-1):
            start_node, end_node = self.path[i], self.path[i+1]
            
            edge = self.create_edge(start_node, end_node)
            self.play(Create(edge), run_time=self.edge_creation_time)
            
            if self.node_fade_time > 0:
                next_node = self.create_node(end_node)
                self.play(FadeIn(next_node), run_time=self.node_fade_time)
        
        self.wait_for_wiggle_period()

class MultiPathAnimation(BaseGraphScene):
    """Scene for animating multiple paths with different colors and fade effects."""
    def __init__(
        self,
        paths,
        path_colors,
        edge_creation_time=0.5,
        node_fade_time=0.3,
        path_fade_time=1.0,
        path_display_time=1.0,
        fade_previous=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.paths = paths
        self.path_colors = path_colors
        self.edge_creation_time = edge_creation_time
        self.node_fade_time = node_fade_time
        self.path_fade_time = path_fade_time
        self.path_display_time = path_display_time
        self.fade_previous = fade_previous
        
        if len(self.path_colors) < len(self.paths):
            self.path_colors.extend([WHITE] * (len(self.paths) - len(self.path_colors)))



    def get_number_updater(self, start_node, end_node):
        """Create an updater function for edge number position."""
        def updater(mob, dt):
            # Get current positions of edge endpoints
            start_pos = self.get_node_position(start_node)
            end_pos = self.get_node_position(end_node)
            
            # Calculate midpoint
            mid_point = (start_pos + end_pos) / 2
            
            # Calculate perpendicular offset
            edge_vector = end_pos - start_pos
            perpendicular = np.array([-edge_vector[1], edge_vector[0], 0])
            perpendicular = perpendicular / np.linalg.norm(perpendicular) * 0.3
            
            # Update number position
            mob.move_to(mid_point + perpendicular)
        return updater

    def create_edge_number(self, start_node, end_node, number, color):
        """Create a number label for an edge with position updates."""
        # Get initial positions
        start_pos = self.get_node_position(start_node)
        end_pos = self.get_node_position(end_node)
        
        # Calculate initial position
        mid_point = (start_pos + end_pos) / 2
        edge_vector = end_pos - start_pos
        perpendicular = np.array([-edge_vector[1], edge_vector[0], 0])
        perpendicular = perpendicular / np.linalg.norm(perpendicular) * 0.3
        
        # Create and position the text
        number_label = Text(str(number), color=color, font_size=24)
        number_label.move_to(mid_point + perpendicular)
        
        # Add the updater
        number_label.add_updater(self.get_number_updater(start_node, end_node))
        
        return number_label



    def animate_single_path(self, path, color):
        """Animate a single path with nodes and edges."""
        nodes = []
        edges = []
        numbers = []
        
        if self.node_fade_time > 0:    
            current_node = self.create_node(path[0], color=color)
            self.play(FadeIn(current_node))
            nodes.append(current_node)
            
        for i in range(len(path)-1):
            start_node, end_node = path[i], path[i+1]
            
            # Create and animate edge
            edge = self.create_edge(start_node, end_node, color=color)
            
            # Create edge number with node references for updating
            number = self.create_edge_number(
                start_node,
                end_node,
                i + 1,  # Start counting from 1
                color
            )
            
            # Animate edge and number together
            self.play(
                Create(edge),
                FadeIn(number),
                run_time=self.edge_creation_time
            )
            
            edges.append(edge)
            numbers.append(number)
            
            if self.node_fade_time > 0:
                next_node = self.create_node(end_node, color=color)
                self.play(FadeIn(next_node), run_time=self.node_fade_time)
                nodes.append(next_node)
        
        return nodes, edges, numbers

    def construct(self):
        self.setup_graph()
        
        previous_nodes = None
        previous_edges = None
        previous_numbers = None
        
        for path, color in zip(self.paths, self.path_colors):
            # If we have previous elements and want to fade them
            if previous_edges is not None and self.fade_previous:
                fade_animations = []
                if previous_nodes:
                    fade_animations.extend(FadeOut(node) for node in previous_nodes)
                if previous_numbers:
                    fade_animations.extend(FadeOut(number) for number in previous_numbers)
                fade_animations.extend(FadeOut(edge) for edge in previous_edges)
                self.play(*fade_animations, run_time=self.path_fade_time)
            
            # Animate current path
            current_nodes, current_edges, current_numbers = self.animate_single_path(path, color)
            
            # Wait for display time
            self.wait(self.path_display_time)
            
            # Store current elements for next iteration
            if self.fade_previous:
                previous_nodes = current_nodes
                previous_edges = current_edges
                previous_numbers = current_numbers
            
        # Fade out the last path if needed
        if self.fade_previous and previous_edges is not None:
            fade_animations = []
            if previous_nodes:
                fade_animations.extend(FadeOut(node) for node in previous_nodes)
            if previous_numbers:
                fade_animations.extend(FadeOut(number) for number in previous_numbers)
            fade_animations.extend(FadeOut(edge) for edge in previous_edges)
            self.play(*fade_animations, run_time=self.path_fade_time)
        
        self.wait_for_wiggle_period()

class EdgeGrowth(BaseGraphScene):
    """Scene for animating the growth of multiple edges simultaneously."""
    
    def __init__(self, edges, edge_creation_time=0.5, n_groups=1, **kwargs):
        super().__init__(**kwargs)
        self.edges_to_create = edges
        self.edge_creation_time = edge_creation_time
        self.n_groups = n_groups

    def construct(self):
        self.setup_graph()
        edges_mobjects = [self.create_edge(u, v) for u, v in self.edges_to_create]

        # Split edges into groups
        default_rng(self.seed).shuffle(edges_mobjects)
        for group_idx in range(self.n_groups):
            group = edges_mobjects[group_idx::self.n_groups]
            self.play(
                *[Create(edge) for edge in group],
                run_time=self.edge_creation_time,
            )
        self.wait_for_wiggle_period()

class ClusteringCoefficientAnimation(BaseGraphScene):
    """Scene for animating local clustering coefficient calculations."""
    
    def __init__(
        self,
        nodes_to_analyze,
        center_color="#FF5353",
        neighbor_color='#5E95FF',
        center_node_fade_time=0.5,
        neighbor_node_fade_time=0.3,
        edge_creation_time=0.5,
        center_edge_width=1.5,
        neighbor_edge_width=2.5,
        display_time=0.5,
        fade_time=0.6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nodes_to_analyze = nodes_to_analyze
        self.center_color = center_color
        self.neighbor_color = neighbor_color
        self.center_node_fade_time = center_node_fade_time
        self.neighbor_node_fade_time = neighbor_node_fade_time
        self.edge_creation_time = edge_creation_time
        self.center_edge_width = center_edge_width
        self.neighbor_edge_width = neighbor_edge_width
        self.display_time = display_time
        self.fade_time = fade_time

    def get_cluster_elements(self, center_node):
        """Get all nodes and edges in the local cluster."""
        # Get neighbor nodes
        neighbors = list(self.G.neighbors(center_node))
        
        # Get edges from center to neighbors
        center_edges = [(center_node, n) for n in neighbors]
        
        # Get edges between neighbors
        neighbor_edges = [
            (u, v) for u in neighbors for v in neighbors 
            if u < v and self.G.has_edge(u, v)
        ]
        
        return neighbors, center_edges, neighbor_edges

    def animate_cluster(self, center_node):
        """Animate the analysis of a single node's clustering coefficient."""
        elements = []  # Track all elements for cleanup
        
        # 1. Fade in center node
        center = self.create_node(center_node, color=self.center_color)
        self.play(FadeIn(center), run_time=self.center_node_fade_time)
        elements.append(center)
        
        # Get cluster elements
        neighbors, center_edges, neighbor_edges = self.get_cluster_elements(center_node)
        
        # 2. Grow edges from center node
        center_edge_mobs = []
        for start, end in center_edges:
            edge = self.create_edge(start, end, color=self.center_color, width=self.center_edge_width)
            center_edge_mobs.append(edge)
            elements.append(edge)
        self.play(
            *[Create(edge) for edge in center_edge_mobs],
            run_time=self.edge_creation_time
        )
            
        
        # 3. Fade in neighbor nodes
        neighbor_mobs = []
        for node in neighbors:
            neighbor = self.create_node(node, color=self.neighbor_color)
            neighbor_mobs.append(neighbor)
            elements.append(neighbor)
        self.play(*[FadeIn(neighbor) for neighbor in neighbor_mobs], run_time = self.neighbor_node_fade_time)
        

        # 4. Fade in edges between neighbors
        neighbor_edge_mobs = []
        for start, end in neighbor_edges:
            edge = self.create_edge(start, end, color=self.neighbor_color, width=self.neighbor_edge_width)
            neighbor_edge_mobs.append(edge)
            elements.append(edge)
        if len(neighbor_edge_mobs) > 0:
            self.play(*[Create(edge) for edge in neighbor_edge_mobs], run_time=self.edge_creation_time) # If there are edges to display

        
        # Wait to display the complete cluster
        self.wait(self.display_time)
        
        return elements

    def construct(self):
        self.setup_graph()
        
        for i, node in enumerate(self.nodes_to_analyze):
            # Animate current cluster
            elements = self.animate_cluster(node)
            
            # Fade out everything if not the last node
            if i < len(self.nodes_to_analyze) - 1:
                self.play(
                    *[FadeOut(elem) for elem in elements],
                    run_time=self.fade_time
                )
        
        # Fade out the last cluster
        self.play(
            *[FadeOut(elem) for elem in elements],
            run_time=self.fade_time
        )
        
        self.wait_for_wiggle_period()



class MultipleCusters(BaseGraphScene):
    def __init__(self, graph, pos, center_nodes, edge_width=1, cluster_fade_time=0.5, cluster_display_time=1, cmap=plt.cm.cool, **kwargs):
        super().__init__(graph, pos, **kwargs)
        self.center_nodes = center_nodes
        self.cluster_fade_time = cluster_fade_time
        self.cluster_display_time = cluster_display_time
        self.cmap = cmap
        self.edge_width = edge_width

    def construct(self):
        self.setup_graph()
        props = np.linspace(0, 1, len(self.center_nodes))
        rng = default_rng(self.seed)
        rng.shuffle(props)
        colors = [ManimColor(self.cmap(i)) for i in props]

        for center_node, color in zip(self.center_nodes, colors):
            nodes, edges = get_local_cluster(self.G, center_node)
            nodes_mobs = [self.create_node(node, color=color) for node in nodes]
            edges_mobs = [self.create_edge(u, v, color=color, width=self.edge_width) for u, v in edges]
            self.play(
                *[FadeIn(node) for node in nodes_mobs],
                *[FadeIn(edge) for edge in edges_mobs],
                run_time=self.cluster_fade_time
            )
            self.wait(self.cluster_display_time)
            self.play(
                *[FadeOut(node) for node in nodes_mobs],
                *[FadeOut(edge) for edge in edges_mobs],
                run_time=self.cluster_fade_time
            )
        self.wait_for_wiggle_period()


class NodeDegreeAnimation(BaseGraphScene):
    '''
        Scene for animating the degree of individual nodes.
    '''
    def __init__(self, graph, pos, center_nodes, edge_width=1, fade_time=0.5, display_time=1, cmap=plt.cm.cool, **kwargs):
        super().__init__(graph, pos, **kwargs)
        self.center_nodes = center_nodes
        self.fade_time = fade_time
        self.display_time = display_time
        self.cmap = cmap
        self.edge_width = edge_width

    def construct(self):
        self.setup_graph()
        props = np.linspace(0, 1, len(self.center_nodes))
        rng = default_rng(self.seed)
        rng.shuffle(props)
        colors = [ManimColor(self.cmap(i)) for i in props]

        for center_node, color in zip(self.center_nodes, colors):
            node_mob = self.create_node(center_node, color=color)
            edges = [(center_node, v) for v in self.G.neighbors(center_node)]
            edges_mobs = [self.create_edge(u, v, color=color, width=self.edge_width) for u, v in edges]
            self.play(
                FadeIn(node_mob),
                *[FadeIn(edge) for edge in edges_mobs],
                run_time=self.fade_time
            )
            self.wait(self.display_time)
            self.play(
                FadeOut(node_mob),
                *[FadeOut(edge) for edge in edges_mobs],
                run_time=self.fade_time
            )
        self.wait_for_wiggle_period()

class ScalingNodes(BaseGraphScene):
    def __init__(self, graph, pos, initial_radius, min_radius, max_radius, duration=4, **kwargs):
        super().__init__(graph, pos, node_radius=initial_radius, **kwargs)
        self.initial_radius = initial_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.duration = duration

    def construct(self):
        self.setup_graph()
        
        # Create nodes with initial radius
        nodes = [self.create_node(node, radius=self.initial_radius) for node in self.G.nodes()]
        edges = [self.create_edge(u, v) for u, v in self.G.edges()]
        self.add(*edges)
        self.add(*nodes)
        
        # Animate scaling of nodes by degree
        anims = []
        for node in nodes:
            node_id = list(self.G.nodes())[nodes.index(node)]
            degree = self.G.degree[node_id]
            max_degree = max(dict(self.G.degree).values())
            target_radius = self.min_radius + (self.max_radius - self.min_radius) * (degree / max_degree)
            anims.append(node.animate.scale(target_radius / self.initial_radius))
        
        self.play(*anims, run_time=self.duration)
        self.wait_for_wiggle_period()


class NodeFailure(BaseGraphScene):
    def __init__(self, graph, pos, failure_nodes, initial_wait=2, fade_time=1,display_time=1.25, between_wait=1, failure_color=RED, **kwargs):
        super().__init__(graph, pos, **kwargs)
        self.failure_nodes = failure_nodes
        self.fade_time = fade_time
        self.failure_color = failure_color
        self.display_time = display_time
        self.initial_wait = initial_wait
        self.between_wait = between_wait

    def construct(self):
        self.setup_graph()
        nodes = [self.create_node(node) for node in self.G.nodes()]
        persistent_edges = [self.create_edge(u, v) for u, v in self.G.edges() if u not in self.failure_nodes and v not in self.failure_nodes]
        failed_edges = {node: [] for node in self.failure_nodes}
        for u, v in self.G.edges():
            if u in self.failure_nodes:
                failed_edges[u].append(self.create_edge(u, v))
            if v in self.failure_nodes:
                failed_edges[v].append(self.create_edge(u, v))


        self.add(*persistent_edges)
        
        for node in self.failure_nodes:
            self.add(*failed_edges[node])

        self.add(*nodes)

        self.wait(self.initial_wait)
        for failed_node in self.failure_nodes:
            failed_edge_mobs = failed_edges[failed_node]
    
            # First change the color of node and edges connected to it
            self.play(
                nodes[failed_node].animate.set_color(self.failure_color),
                *[edge.animate.set_color(self.failure_color) for edge in failed_edge_mobs],
                run_time=self.fade_time
            )
            self.wait(self.display_time)

            # Then fade out the node and edges
            self.play(
                FadeOut(nodes[failed_node]),
                *[Uncreate(edge) for edge in failed_edge_mobs],
                run_time=self.fade_time
            )
            self.wait(self.between_wait)

        self.wait_for_wiggle_period()