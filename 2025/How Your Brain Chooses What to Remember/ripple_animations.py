import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d
import cmasher
from pathlib import Path
import os
from artemutils.colormaps import get_colormap

def create_population_voltage_matrix(spike_times_dict, t_start=None, t_end=None, dt=0.1, sigma=10):
    all_spikes = np.concatenate([spikes for spikes in spike_times_dict.values() if len(spikes) > 0])
    t_start = t_start if t_start is not None else np.floor(np.min(all_spikes))
    t_end = t_end if t_end is not None else np.ceil(np.max(all_spikes))
    
    
    time = np.arange(t_start, t_end, dt)
    sigma_idx = sigma / dt # Convert sigma from ms to array indices
    
    # Initialize voltage matrix
    neuron_ids = list(spike_times_dict.keys())
    voltage_matrix = np.zeros((len(neuron_ids), len(time)))
    
    # Create voltage trace for each neuron
    for i, nid in enumerate(neuron_ids):
        spike_times = spike_times_dict[nid]
        if len(spike_times) > 0:  # Skip empty spike trains
            spike_train = np.zeros_like(time)
            spike_indices = np.searchsorted(time, spike_times)
            valid_indices = (spike_indices >= 0) & (spike_indices < len(time))
            spike_train[spike_indices[valid_indices]] = 1
            
            # Apply Gaussian smoothing
            voltage_matrix[i] = gaussian_filter1d(spike_train, sigma_idx)
    
    # Normalize each neuron's trace to [0,1] range
    max_vals = np.maximum(np.max(voltage_matrix, axis=1, keepdims=True), 1e-10)
    voltage_matrix = voltage_matrix / max_vals
    return time, voltage_matrix, neuron_ids




def create_network_visualization(graph, pos, voltages, cell_type, save_folder, node_color='white', save_edges=True):
    save_folder_full = Path(f'/Users/artemkirsanov/YouTube/SWR memory selection/Code/animations/{save_folder}')
    save_folder_full.mkdir(parents=True, exist_ok=True)
    
    # Calculate fixed axis limits from pos dictionary
    pos_array = np.array(list(pos.values()))
    x_min, y_min = pos_array.min(axis=0) - 0.1  # Add padding
    x_max, y_max = pos_array.max(axis=0) + 0.1
    
    def setup_axes():
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=200)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axis('off')
        return fig, ax
    

    
    # Save edges if requested
    if save_edges:
        fig, ax = setup_axes()
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='gray', width=0.25)
        plt.savefig(save_folder_full / 'edges.png', dpi=200, transparent=True)
        plt.close()
    
    
    

    # Draw nodes for the current cell type
    fig, ax = setup_axes()
    nodelist = [n for n, attr in graph.nodes(data=True) if attr['cell_type'] == cell_type]
    nodes = nx.draw_networkx_nodes(graph, pos,
                                    nodelist=nodelist,
                                    node_color=node_color,
                                    node_size=270,
                                    ax=ax,
                                    node_shape='^' if cell_type == 'PC' else 'o')
    
    # Save matte
    plt.savefig(save_folder_full / f'{cell_type}_matte.png', dpi=200, transparent=True)
    
    # Animation update function
    def update(frame):
        nodes.set_alpha(voltages[:, frame])
        return nodes,
    
    ani = FuncAnimation(fig, update,
                        frames=tqdm(np.arange(voltages.shape[1])),
                        blit=True,
                        interval=30)
    
    ani.save(save_folder_full / f'{cell_type}_activity.mov', dpi=200)
    plt.close()
    print(f'Saved {cell_type} animation')

def animate_SWR_timeseries(t, lfp):
    fig, ax = plt.subplots(1, 1, figsize=(16, 3), dpi=300)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.plot(t, lfp, color='black', lw=1) # This is just to set the axis limits


    line = ax.plot([], [], color='white', lw=1, solid_capstyle='round')[0]
    def update(frame):
        line.set_data(t[:frame], lfp[:frame])
        return line,    
    ani = FuncAnimation(fig, update,
                        frames=tqdm(np.arange(len(t))),
                        blit=True,
                        interval=30)
    ani.save('/Users/artemkirsanov/YouTube/SWR memory selection/Code/animations/SWR_timeseries.mov')
    print('Saved LFP animation')
    plt.close()

def animate_ca3net_ripple():
    # Load data from example seed provided in the repository
    replay = pickle.load(open('/Users/artemkirsanov/YouTube/SWR memory selection/Code/ca3net/files/replay_12345.pkl', 'rb'))
    lfp = pickle.load(open('/Users/artemkirsanov/YouTube/SWR memory selection/Code/ca3net/files/LFP_12345.pkl', 'rb'))
    
    # PC spikes
    PC = pickle.load(open('/Users/artemkirsanov/YouTube/SWR memory selection/Code/ca3net/files/PC_spikes_12345.pkl', 'rb'))
    PC_spikes = [PC['spike_times'][PC['spiking_neurons'] == k] for k in np.unique(PC['spiking_neurons'])]

    # BC spikes (inhibitory)
    BC=pickle.load(open('/Users/artemkirsanov/YouTube/SWR memory selection/Code/ca3net/files/BC_spikes_12345.pkl', 'rb'))
    BC_spikes = [BC['spike_times'][BC['spiking_neurons'] == k] for k in np.unique(BC['spiking_neurons'])]

    # --- Selecting one example SWR event
    bounds = list(replay['replay_results'].keys())[1]
    offset = 150 # 150 ms before and after the event 
    replay_start, replay_end = bounds[0] - offset, bounds[1] + offset


    # --- Filtering spikes that occur within the replay event
    PC_spikes_filtered = [spike[spike > replay_start] for spike in PC_spikes]
    PC_spikes_filtered = [spike[spike < replay_end] for spike in PC_spikes_filtered]
    BC_spikes_filtered = [spike[spike > replay_start] for spike in BC_spikes]
    BC_spikes_filtered = [spike[spike < replay_end] for spike in BC_spikes_filtered]

    # --- Selecting neurons that are active during the replay event
    PC_active = np.array([k for k in range(len(PC_spikes_filtered)) if len(PC_spikes_filtered[k]) > 0])
    BC_active = np.array([k for k in range(len(BC_spikes_filtered)) if len(BC_spikes_filtered[k]) > 0])

    np.random.seed(42)
    selected_neurons_PC = PC_active[sorted(np.random.choice(np.arange(len(PC_active)), 50, replace=False))]
    selected_neurons_BC = BC_active[sorted(np.random.choice(np.arange(len(BC_active)), 14, replace=False))]
    selected_spikes_PC = {k: PC_spikes_filtered[k] for k in selected_neurons_PC}
    selected_spikes_BC = {k: BC_spikes_filtered[k] for k in selected_neurons_BC}

    # --- Create a graph
    fake_graph = nx.watts_strogatz_graph(
        len(selected_neurons_BC) + len(selected_neurons_PC),
        p=0.2,
        k=15,
        seed=42
    )

    # For each node in the graph, compute the degree and assign nodes with highest degree to inhibitory neurons
    degrees = dict(fake_graph.degree())
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    inhibitory_nodes = [node for node, _ in sorted_degrees[:len(selected_neurons_BC)]]
    neuron_types = {node: 'BC' if node in inhibitory_nodes else 'PC' for node in fake_graph.nodes()}
    nx.set_node_attributes(fake_graph, neuron_types, 'cell_type')

    # Set node names to neuron IDs
    node_names = {i: f'PC_{k}' for i, k in enumerate(selected_neurons_PC)}
    node_names.update({i + len(selected_neurons_PC): f'BC_{k}' for i, k in enumerate(selected_neurons_BC)})
    fake_graph = nx.relabel_nodes(fake_graph, node_names)

    # --- Create voltage traces
    time, voltages_PC, _ = create_population_voltage_matrix(selected_spikes_PC, t_start=replay_start, t_end=replay_end, dt=0.25, sigma=0.5)
    time, voltages_BC, _ = create_population_voltage_matrix(selected_spikes_BC, t_start=replay_start, t_end=replay_end, dt=0.25, sigma=0.5)

    # --- Create animations
    pos = nx.spring_layout(fake_graph, seed=42)
    # create_network_visualization(fake_graph, pos, voltages_PC, 'PC', 'ripple_small', node_color='white') White PC nodes
    # create_network_visualization(fake_graph, pos, voltages_BC, 'BC', 'ripple_small', node_color='white', save_edges=False) White BC nodes


    # Colored PC nodes 
    cmap = get_colormap('coolors-1')
    pyr_nodes = [n for n, attr in fake_graph.nodes(data=True) if attr['cell_type'] == 'PC']
    node_colors = [cmap(i / len(pyr_nodes)) for i in range(len(pyr_nodes))]
    create_network_visualization(fake_graph, pos, voltages_PC, 'PC', 'ripple_small_color', node_color=node_colors)






    # --- Create LFP animation
    # time_lfp = lfp['t']
    # # Interpolate LFP to match the time resolution of the voltage traces
    # lfp_interp = np.interp(time, time_lfp, lfp['LFP'])
    # animate_SWR_timeseries(time, lfp_interp)



if __name__ == '__main__':
    animate_ca3net_ripple()