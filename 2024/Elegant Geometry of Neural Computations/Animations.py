import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import numpy as np
import os
from tqdm import tqdm
import cmasher
import seaborn as sns
from NeuronModel import NeuronModel
from pathlib import Path
from termcolor import colored
from scipy.stats import qmc

VOLTAGE_LIMITS = [-90, 20]
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)

def latin_hypercube_sampling_2d(n_samples, x_bounds, y_bounds, seed=None):
    """
    Generate Latin Hypercube samples in 2D space with custom bounds using scipy.
    
    Parameters:
    -----------
    n_samples : int
        Number of points to sample
    x_bounds : tuple
        (min, max) bounds for x dimension
    y_bounds : tuple
        (min, max) bounds for y dimension
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (n_samples, 2) containing the sampled points
    """
    # Create the sampler
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    
    # Generate samples in [0, 1]^2 space
    samples = sampler.random(n=n_samples)
    
    # Scale samples to desired bounds
    bounds = np.array([x_bounds, y_bounds])
    samples_scaled = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
    
    return samples_scaled


def resample_curve(V, n, n_points=1000):
    """
    Resample V,n arrays to have points at equal arc length intervals
    
    Parameters:
    -----------
    V, n : array-like
        Original trajectory coordinates
    n_points : int
        Number of points in resampled curve
        
    Returns:
    --------
    V_new, n_new : arrays
        Resampled trajectory coordinates
    """
    # Calculate cumulative distance along the curve
    points = np.column_stack((V, n))
    diff = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
    cumulative_length = np.concatenate(([0], np.cumsum(segment_lengths)))
    total_length = cumulative_length[-1]
    
    # Generate equal-spaced points along the curve length
    equal_spaces = np.linspace(0, total_length, n_points)
    
    # Interpolate to get new points
    V_new = np.interp(equal_spaces, cumulative_length, V)
    n_new = np.interp(equal_spaces, cumulative_length, n)
    
    return V_new, n_new

def make_segments(x,y):
    ''' Create segments for LineCollection '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)

def cmap_from_hex_list(color_positions):
    """
    Create a colormap from a list of (hex_color, position) tuples.
    
    Args:
        color_positions: List of tuples [(hex_color, position)], where position is between 0 and 1
    
    Returns:
        LinearSegmentedColormap object
    """
    # Convert hex to RGB
    colors_rgb = [(int(c.lstrip('#')[0:2], 16) / 255,
                  int(c.lstrip('#')[2:4], 16) / 255,
                  int(c.lstrip('#')[4:6], 16) / 255) for c, _ in color_positions]
    
    positions = [pos for _, pos in color_positions]
    # Create the colormap
    return LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors_rgb)))


def make_folder(name, root_dir='/Users/artemkirsanov/YouTube/Bifurcations/Assets/Phase portrait animations'):
    path = Path(root_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path

    
def setup_phase_portait_ax(show_ax=False, dpi=300):
    '''
        Set up the phase portrait axes with a black background and white gridlines.

        Parameters:
        ----------
        show_ax : bool
            Whether to show the axes labels and ticks. If False, the labels and ticks will be black on a black background (useful for overlaying animations).
        Returns:
        ----------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
    '''
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(10,10), dpi=dpi)
    fig.set_facecolor('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if show_ax:
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.tick_params(colors='white', labelsize=20)
    else:
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(colors='black', labelsize=20)

    ax.set_facecolor('black')
    if show_ax:
        ax.set_xlabel('Voltage', color='white', fontsize=22)
        ax.set_ylabel('Frac. of potassium channels', color='white', fontsize=22)
    else:
        ax.set_xlabel('Voltage', color='black', fontsize=22)
        ax.set_ylabel('Frac. of potassium channels', color='black', fontsize=22)

    ax.set_xlim(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1])
    ax.set_ylim(-0.025, 1.025)
    return fig, ax


def setup_timeseries_ax(show_ax=False, kind='V'):
    '''
        Set up the time series axes with a black background and white gridlines.

        Parameters:
        ----------
        show_ax : bool
            Whether to show the axes labels and ticks. If False, the labels and ticks will be black on a black background (useful for overlaying animations).
        kind : str
            The type of time series to plot ('V' or 'n').
        Returns:
        ----------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
    '''
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(10,4), dpi=300)
    fig.set_facecolor('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_ax:
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.tick_params(colors='white', labelsize=20)
    else:
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(colors='black', labelsize=20)

    ax.set_facecolor('black')
    ax.set_xlabel('Time, s', color='white' if show_ax else 'black', fontsize=22)
    if kind == 'V':
        ax.set_ylabel('$V$, mV', color='white' if show_ax else 'black', fontsize=22)
        ax.set_ylim(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1])
    else:
        ax.set_ylabel('$n$', color='white' if show_ax else 'black', fontsize=22)
        ax.set_ylim(-0.025, 1.025)

    fig.tight_layout() # Adjust layout to make room for labels
    return fig, ax


def setup_animation_folder(base_dir, prefix="phase_portrait"):
    '''
        Create a new folder for saving animations with the current date and time as the suffix.
    '''
    from datetime import datetime
    folder_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def animate_vector_field(neuron, currents, save_folder, vector_field_resolution=30):
    '''
        Generate a vector field animation for the given neuron model and current values.

        Parameters:
        ----------
        neuron : NeuronModel
            The neuron model to use for the vector field (must have a dALLdt method).
        currents : array-like
            The current values to animate. If all currents are the same, the vector field will be static.
        save_folder : str
            The folder in which to save the animation.
        vector_field_resolution : int
            The number of points in each dimension to use for the vector field
    '''
    fig, ax = setup_phase_portait_ax()
    
    x = np.linspace(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1],  vector_field_resolution)
    y = np.linspace(0, 1, vector_field_resolution)
    X, Y = np.meshgrid(x, y)
    cmap = cmasher.get_sub_cmap(plt.cm.Greys_r, 0.15, 0.8)
    quiver = None
    def update(I_ext):
        nonlocal quiver
        if quiver is not None:
            quiver.remove()
        
        dxdt, dydt = neuron.dALLdt([X, Y], 0, lambda t: I_ext)
        dydt_amplified = dydt * 100
        norm = np.sqrt(dxdt**2 + dydt_amplified**2)
        dxdt_norm = dxdt / norm
        dydt_norm = dydt_amplified / norm
        
        quiver = ax.quiver(X, Y, dxdt_norm, dydt_norm, norm,
                          alpha=1, headwidth=4, headlength=3,
                          headaxislength=2, pivot='mid', cmap=cmap)
        return quiver,
    
    if np.all(np.isclose(currents, currents[0])):
        # If all currents are the same, save a static image
        update(currents[0])
        save_path = os.path.join(save_folder, 'vector_field.png')
        fig.savefig(save_path, dpi=300)

    else:
        anim = FuncAnimation(fig, update, frames=tqdm(currents),
                            interval=30, blit=True)
        save_path = os.path.join(save_folder, 'vector_field.mp4')
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved vector field animation to {save_path}", 'green'))


def animate_streamlines(neuron, currents, save_folder, vector_field_resolution=50, density=1.5):
    """Generate streamlines animation"""
    fig, ax = setup_phase_portait_ax()

    x = np.linspace(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1], vector_field_resolution)
    y = np.linspace(0, 1, vector_field_resolution)
    X, Y = np.meshgrid(x, y)
    dxdt, dydt = neuron.dALLdt([X, Y], 0, lambda t: currents[0])
    
    cmap = cmasher.get_sub_cmap(plt.cm.Greys_r, 0.15, 0.8)

    def update(I_ext):
        ax.clear()
        dxdt, dydt = neuron.dALLdt([X, Y], 0, lambda t: I_ext)
        norm = np.sqrt(dxdt**2 + dydt**2)
        
        stream_lines = ax.streamplot(X, Y, dxdt, dydt, 
                                   color=norm,
                                   linewidth=1,
                                    density=density,
                                   broken_streamlines=False,
                                   cmap=cmap, 
                                   arrowsize=0)
        return stream_lines.lines,


    if np.all(np.isclose(currents, currents[0])):
        # If all currents are the same, save a static image
        update(currents[0])
        save_path = os.path.join(save_folder, 'streamlines.png')
        fig.savefig(save_path, dpi=300)
    else:
        anim = FuncAnimation(fig, update, frames=tqdm(currents),
                            interval=30, blit=False)
        save_path = os.path.join(save_folder, 'streamlines.mp4')

        anim.save(save_path, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved streamlines animation to {save_path}", 'green'))


def animate_limit_cycle(neuron, currents, save_folder, lw=3, color='white', ls='-', alpha=1, suffix="", T_max=200):
    '''
        Generate an animation of the limit cycle for the given neuron model and current values. If the current values are the same, save a static image.

    '''
    fig, ax = setup_phase_portait_ax()
    V, n = neuron.find_limit_cycle(currents[0], T_max=T_max)
    line = ax.plot(V, n, color=color, lw=lw, ls=ls, alpha=alpha)[0]

    def update(I_ext):
        V, n = neuron.find_limit_cycle(I_ext,T_max=T_max)
        line.set_data(V, n)
        return line,

    if np.all(np.isclose(currents, currents[0])):
        # If all currents are the same, save a static image
        update(currents[0])
        save_path = os.path.join(save_folder, f'limit_cycle_{suffix}.png')
        fig.savefig(save_path, dpi=300)
    
    else:
        anim = FuncAnimation(fig, update, frames=tqdm(currents),
                            interval=30, blit=True)
        save_path = os.path.join(save_folder, f'limit_cycle_{suffix}.mp4')
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved limit cycle animation to {save_path}", 'green'))


def animate_limit_cycle_time_series(neuron, currents, save_folder, lw=3, color='white', 
                                  suffix="", n_cycles=7):
    '''
    Generate a smooth animation of voltage oscillations with phase alignment
    '''
    # First get all traces to determine common time axis
    all_traces = []
    max_len = 0
    
    for I in currents:
        t, V = neuron.find_aligned_limit_cycle(I, n_cycles=n_cycles)
        if t is not None and V is not None:
            all_traces.append((t, V))
            max_len = max(max_len, len(t))
    
    # Create common time axis
    dt = all_traces[0][0][1] - all_traces[0][0][0]
    t_common = np.arange(max_len) * dt
    
    # Interpolate all traces to common time axis
    V_interpolated = []
    for t, V in all_traces:
        if len(V) < max_len:
            # Extend the trace by repeating it
            n_repeats = int(np.ceil(max_len / len(V)))
            V_extended = np.tile(V, n_repeats)[:max_len]
            V_interpolated.append(V_extended)
        else:
            V_interpolated.append(V[:max_len])
    
    # Save axes
    fig, ax = setup_timeseries_ax(kind='V', show_ax=True)
    ax.set_xlim(t_common[0], t_common[-1])
    ax.set_ylim(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1])    
    save_path = os.path.join(save_folder, f'limit_cycle_timeseries_axes_{suffix}.png')
    fig.savefig(save_path, dpi=300)
    plt.close()
    
    # Animate time series
    fig, ax = setup_timeseries_ax(kind='V', show_ax=False)
    ax.set_xlim(t_common[0], t_common[-1])
    ax.set_ylim(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1])
    
    line = ax.plot([], [], color=color, lw=lw)[0]
    
    def update(frame):
        i = frame
        line.set_data(t_common, V_interpolated[i])
        return line,
    
    if np.all(np.isclose(currents, currents[0])):
        update(0)
        save_path = os.path.join(save_folder, f'limit_cycle_timeseries_{suffix}.png')
        fig.savefig(save_path, dpi=300)
    else:
        anim = FuncAnimation(fig, update, frames=tqdm(range(len(currents))),
                           interval=30, blit=True)
        save_path = os.path.join(save_folder, f'limit_cycle_timeseries_{suffix}.mp4')
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=300)
    
    plt.close()
    print(colored(f"Saved limit cycle timeseries animation to {save_path}", 'green'))


def animate_unstable_limit_cycle(neuron, currents, save_folder, lw=3, color='white', ls='-', alpha=1, suffix="", T_max=1000):
    fig, ax = setup_phase_portait_ax()
    V, n = neuron.find_unstable_limit_cycle(currents[0], T_max=T_max)
    line = ax.plot(V, n, color=color, lw=lw, ls=ls, alpha=alpha)[0]

    def update(I_ext):
        try:
            V, n = neuron.find_unstable_limit_cycle(I_ext, T_max=T_max)
            print(f"Current: {I_ext}, Length: {len(V)}")
            line.set_data(V, n)
        except:
            line.set_data([], [])
        return line,

    if np.all(np.isclose(currents, currents[0])):
        # If all currents are the same, save a static image
        update(currents[0])
        save_path = os.path.join(save_folder, f'unstable_limit_cycle_{suffix}.png')
        fig.savefig(save_path, dpi=300)
    
    else:
        anim = FuncAnimation(fig, update, frames=tqdm(currents),
                            interval=30, blit=True)
        save_path = os.path.join(save_folder, f'unstable_limit_cycle_{suffix}.mp4')
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved unstable limit cycle animation to {save_path}", 'green'))

def animate_spiking_orbit_subcritical_Hopf(neuron, currents, save_folder, lw=3, color='white', suffix="", T_max=1000):
    fig, ax = setup_phase_portait_ax()
    V, n = neuron.find_spiking_orbit_subcritical_Hopf(currents[0], T_max=T_max)
    line = ax.plot(V, n, color=color, lw=lw)[0]

    def update(I_ext):
        try:
            V, n = neuron.find_spiking_orbit_subcritical_Hopf(I_ext, T_max=T_max)
            line.set_data(V, n)
        except:
            line.set_data([], [])
        return line,

    if np.all(np.isclose(currents, currents[0])):
        # If all currents are the same, save a static image
        update(currents[0])
        save_path = os.path.join(save_folder, f'spiking_orbit_{suffix}.png')
        fig.savefig(save_path, dpi=300)
    
    else:
        anim = FuncAnimation(fig, update, frames=tqdm(currents),
                            interval=30, blit=True)
        save_path = os.path.join(save_folder, f'spiking_orbit_{suffix}.mp4')
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved spiking orbit subcritical Hopf animation to {save_path}", 'green'))


def animate_invariant_circle(neuron, currents, save_folder, lw=3, suffix=""):
    '''
        Generate an animation of the invariant circle for the given neuron model and current values. If the current values are the same, save a static image.

    '''
    fig, ax = setup_phase_portait_ax()
    V, n = neuron.find_invariant_circle(currents[0])

    cmap = cmap_from_hex_list([('FF5555', 0), ('FF5555', 0.4), ('9C55FF', 0.6), ('556CFF', 0.8), ('556CFF', 1)])
    lc = LineCollection(make_segments(V, n), cmap=cmap, lw=lw)
    lc.set_capstyle("round")
    ax.add_collection(lc)

    def update(I_ext):
        try:
            V,n = neuron.find_invariant_circle(I_ext)
            V, n = resample_curve(V, n)
        except:
            V,n = neuron.find_limit_cycle(I_ext, T_max=100)
            lc.set_cmap(cmap_from_hex_list([('FF5555', 0), ('FF5555', 1)]))

        lc.set_array(np.linspace(0, 1, len(V)))
        lc.set_segments(make_segments(V, n))
        return lc,

    if np.all(np.isclose(currents, currents[0])):
        # If all currents are the same, save a static image
        update(currents[0])
        save_path = os.path.join(save_folder, f'invariant_circle_{suffix}.png')
        fig.savefig(save_path, dpi=300)
    
    else:
        anim = FuncAnimation(fig, update, frames=tqdm(currents),
                            interval=30, blit=True)
        save_path = os.path.join(save_folder, f'invariant_circle_{suffix}.mp4')
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved invariant circle animation to {save_path}", 'green'))


def plot_gradient_lines(Vs, ns, save_folder, lw=2.5, to_resample=True, suffix=""):
    '''
        Plot gradient lines on the phase portrait.
    '''
    fig, ax = setup_phase_portait_ax()
    cmap = cmap_from_hex_list([('FF5555', 0), ('FF5555', 0.4), ('9C55FF', 0.6), ('556CFF', 0.8), ('556CFF', 1)])
    for i in range(len(Vs)):
        V, n = Vs[i], ns[i]
        if to_resample:
            V, n = resample_curve(V, n)
        lc = LineCollection(make_segments(V, n), cmap=cmap, lw=lw)
        lc.set_capstyle("round")
        lc.set_array(np.linspace(0, 1, len(V)))
        ax.add_collection(lc)

    save_path = os.path.join(save_folder, f'gradient_lines_{suffix}.png')

    fig.savefig(save_path, dpi=300)
    plt.close()
    print(colored(f"Saved gradient lines to {save_path}", 'green'))


def animate_V_nullcline(neuron, currents, save_folder, color='#ff1c42', lw=3, arrows=False, arrow_density=40, arrow_scale=5):
    '''
        Generate an animation of the V-nullcline for the given neuron model and current values. Optionally, include arrows showing the direction of the flow (saved as a separate animation).
    '''
    x_high_res = np.linspace(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1], 1000) # Voltage values for high resolution nullcline
    x_arrows = np.linspace(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1], arrow_density) # Voltage values for arrows
    y_arrows = neuron.V_nullcline(x_arrows, currents[0]) # n-nullcline values for arrows

    fig, ax = setup_phase_portait_ax()
    if arrows:
        fig_arrows, ax_arrows = setup_phase_portait_ax()
    quiver = None
    line = ax.plot(x_high_res, neuron.V_nullcline(x_high_res, currents[0]), color=color, lw=lw)[0]

    def update(I_ext):
        nonlocal quiver
        if quiver is not None:
            quiver.remove()
        
        if arrows:
            y_arrows = neuron.V_nullcline(x_arrows, I_ext) # n-nullcline values for arrows

            dxdt, dydt = neuron.dALLdt([x_arrows, y_arrows], 0, lambda t: I_ext)
            norm = np.sqrt(dxdt**2 + dydt**2)
            dxdt, dydt = dxdt / norm, dydt / norm
            
            quiver = ax_arrows.quiver(x_arrows, y_arrows, dxdt, dydt*arrow_scale, width=0.005,
                            alpha=1, headwidth=4, headlength=3, 
                            headaxislength=2, pivot='mid', color=color)
        line.set_data(x_high_res, neuron.V_nullcline(x_high_res, I_ext))
        if arrows:
            return line, quiver
        return line,

    if np.all(np.isclose(currents, currents[0])):
        # If all currents are the same, save a static image
        update(currents[0])
        save_path = os.path.join(save_folder, 'v_nullcline.png')
        fig.savefig(save_path, dpi=300)
        if arrows:
            fig_arrows.savefig(save_path.replace('.png', '_arrows.png'), dpi=300)
    else:
        anim_v = FuncAnimation(fig, update, frames=tqdm(currents),
                            interval=30, blit=True)
        save_path = os.path.join(save_folder, 'v_nullcline.mp4')
        anim_v.save(save_path, writer='ffmpeg', fps=30, dpi=300)
        if arrows:
            anim_arrows = FuncAnimation(fig_arrows, update, frames=tqdm(currents),
                                    interval=30, blit=True)
            save_path_arrows = os.path.join(save_folder, 'v_nullcline_arrows.mp4')
            anim_arrows.save(save_path_arrows, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved V-nullcline animation to {save_path}", 'green'))



def animate_n_nullcline(neuron, currents, save_folder, color='#1cffa4', lw=3, arrows=False, arrow_density=40, arrow_scale=5):
    '''
        Generate an animation of the n-nullcline for the given neuron model and current values. Optionally, include arrows showing the direction of the flow (saved as a separate animation).
    '''

    x_high_res = np.linspace(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1], 1000) # Voltage values for high resolution nullcline
    x_arrows = np.linspace(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1], arrow_density) # Voltage values for arrows
    y_arrows = neuron.n_nullcline(x_arrows) # n-nullcline values for arrows

    fig, ax = setup_phase_portait_ax()
    if arrows:
        fig_arrows, ax_arrows = setup_phase_portait_ax()
    
    quiver = None
    line = ax.plot(x_high_res, neuron.n_nullcline(x_high_res), color=color, lw=lw)[0]

    def update(I_ext):
        nonlocal quiver
        if quiver is not None:
            quiver.remove()
        
        if arrows:
            y_arrows = neuron.n_nullcline(x_arrows)
            dxdt, dydt = neuron.dALLdt([x_arrows, y_arrows], 0, lambda t: I_ext)
            norm = np.sqrt(dxdt**2 + dydt**2)
            dxdt, dydt = dxdt / norm, dydt / norm
            
            quiver = ax_arrows.quiver(x_arrows, y_arrows, dxdt, dydt*arrow_scale, width=0.005,
                            alpha=1, headwidth=4, headlength=3,
                            headaxislength=2, pivot='mid', color=color)
        line.set_data(x_high_res, neuron.n_nullcline(x_high_res))
        if arrows:
            return line, quiver
        return line,
    
    if np.all(np.isclose(currents, currents[0])):
        # If all currents are the same, save a static image
        update(currents[0])
        save_path = os.path.join(save_folder, 'n_nullcline.png')
        fig.savefig(save_path, dpi=300)
        if arrows:
            fig_arrows.savefig(save_path.replace('.png', '_arrows.png'), dpi=300)
    else:
        anim_n = FuncAnimation(fig, update, frames=tqdm(currents),
                            interval=30, blit=True)
        save_path = os.path.join(save_folder, 'n_nullcline.mp4')
        anim_n.save(save_path, writer='ffmpeg', fps=30, dpi=300)
        if arrows:
            anim_arrows = FuncAnimation(fig_arrows, update, frames=tqdm(currents),
                                    interval=30, blit=True)
            save_path_arrows = os.path.join(save_folder, 'n_nullcline_arrows.mp4')
            anim_arrows.save(save_path_arrows, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved n-nullcline animation to {save_path}", 'green'))





def animate_equilibrium_points(neuron, currents, save_folder):
    fig, ax = setup_phase_portait_ax()
    v_high_res = np.linspace(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1], 1000)
    n_high_res = np.linspace(0, 1, 1000)
    
    # Create initial scatter plots for each type of point
    stable_points = ax.plot([], [], 'o', color='#556CFF', markersize=12)[0]
    unstable_points = ax.plot([], [], 'o', color='#FF5555', markersize=12)[0]
    saddle_points = ax.plot([], [], 'o', color='#9C55FF', markersize=12)[0]

    def init():
        return stable_points, unstable_points, saddle_points
    def update(I_ext):
        # Find equilibrium points
        eq_points = neuron.find_equlibrium_points(I_ext, [v_high_res[0], v_high_res[-1]])
        
        # Separate points by stability
        stable_v, stable_n = [], []
        unstable_v, unstable_n = [], []
        saddle_v, saddle_n = [], []
        
        for eq_point in eq_points:
            v, n = eq_point['point']
            if eq_point['stability'] == 'stable':
                stable_v.append(v)
                stable_n.append(n)
            elif eq_point['stability'] == 'unstable':
                unstable_v.append(v)
                unstable_n.append(n)
            else:  # saddle
                saddle_v.append(v)
                saddle_n.append(n)
        
        # Update data for each scatter plot
        stable_points.set_data(stable_v, stable_n)
        unstable_points.set_data(unstable_v, unstable_n)
        saddle_points.set_data(saddle_v, saddle_n)
        
        return stable_points, unstable_points, saddle_points
    
    if np.all(np.isclose(currents, currents[0])):
        # If all currents are the same, save a static image
        update(currents[0])
        save_path = os.path.join(save_folder, 'equilibrium_points.png')
        fig.savefig(save_path, dpi=300)
    else:
        anim = FuncAnimation(fig, update, frames=tqdm(currents), 
                            init_func=init, interval=30, blit=True)
        save_path = os.path.join(save_folder, 'equilibrium_points.mp4')
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved equilibrium points animation to {save_path}", 'green'))



def plot_saddle_eigenvectors(neuron, I_ext, save_folder):
    # First find equilibrium points
    equilibria = neuron.find_equlibrium_points(I_ext, [VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1]])
    
    # Find the saddle point
    saddle_point = None
    for eq in equilibria:
        if eq['stability'] == 'saddle':
            saddle_point = eq
            break
    
    if saddle_point is None:
        raise ValueError("No saddle point found. This might not be a saddle-node bifurcation case.")
    
    eig_vector_1 = saddle_point['jacobian'][:,0]
    eig_vector_2 = saddle_point['jacobian'][:,1]

    # Normalize the eigenvectors
    eig_vector_1 = eig_vector_1 / np.linalg.norm(eig_vector_1) * 20
    eig_vector_2 = eig_vector_2 / np.linalg.norm(eig_vector_2) * 20

    # Plot the eigenvectors
    fig, ax = setup_phase_portait_ax()
    ax.plot([saddle_point['point'][0] - eig_vector_1[0], saddle_point['point'][0] + eig_vector_1[0]], [saddle_point['point'][1] - eig_vector_1[1], saddle_point['point'][1] + eig_vector_1[1]], color='white', lw=1, ls='--')
    ax.plot([saddle_point['point'][0]  - eig_vector_2[0], saddle_point['point'][0] + eig_vector_2[0]], [saddle_point['point'][1]  - eig_vector_2[1], saddle_point['point'][1] + eig_vector_2[1]], color='white', lw=1, ls='--')

    save_path = os.path.join(save_folder, 'saddle_eigenvectors.png')
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(colored(f"Saved saddle eigenvectors to {save_path}", 'green'))

def plot_separatrix(neuron, I_ext, save_folder):
    V_sep, n_sep = neuron.find_separatrix(I_ext)
    save_still_trajectory(V_sep, n_sep, 'white', save_folder, 'separatrix')

def animate_separatrix(neuron, currents, save_folder, lw=2, color='white', suffix="", close_curve=False, ls='dashed'):
    fig, ax = setup_phase_portait_ax()
    line = ax.plot([], [], color=color, lw=lw, ls=ls)[0]

    def update(I_ext):
        try:
            V, n = neuron.find_separatrix(I_ext)
            if close_curve: # Close the separatrix curve if needed for paint bucket fill in video editing
                V = np.append(V, V[0])
                n = np.append(n, n[0]) 
            line.set_data(V, n)
        except:
            line.set_data([], []) # If the separatrix is not found, clear the line
        return line,

    if np.all(np.isclose(currents, currents[0])):
        # If all currents are the same, save a static image
        update(currents[0])
        save_path = os.path.join(save_folder, f'separatrix_{suffix}.png')
        fig.savefig(save_path, dpi=300)
    else:
        anim = FuncAnimation(fig, update, frames=tqdm(currents),
                            interval=30, blit=True)
        save_path = os.path.join(save_folder, f'separatrix_{suffix}.mp4')
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved separatrix animation to {save_path}", 'green'))





def animate_color_square(t, values, save_folder, vmin=None, vmax=None, cmap='neuron_voltage', 
                        subsample=1, suffix=""):
    """
    Animate a square changing colors based on a time series of values.
    Creates a clean video with just the colored square for compositing.
    
    Parameters:
    -----------
    t : array-like
        Time points
    values : array-like
        Values to be mapped to colors
    save_folder : str
        Folder to save the animation
    vmin : float or None
        Minimum value for color scaling. If None, uses min(values)
    vmax : float or None
        Maximum value for color scaling. If None, uses max(values)
    cmap : str or matplotlib.colors.Colormap
        Either a colormap name string or a matplotlib colormap object
    subsample : int
        Take every nth point for smoother animation
    suffix : str
        Suffix for the output filename
    """
    # Set up the figure with a transparent background
    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_alpha(0)
    
    # Create axis that fills the entire figure and make it transparent
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Create color normalizer
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Handle colormap input
    if cmap == 'neuron_voltage':
        colormap = cmasher.get_sub_cmap(sns.color_palette("mako", as_cmap=True), 0.1, 1)
    else:
        if isinstance(cmap, str):
            colormap = plt.get_cmap(cmap)
        else:
            colormap = cmap
        
    # Create the square that fills the entire frame
    square = Rectangle((0, 0), 1, 1, facecolor=colormap(norm(values[0])), edgecolor='none')
    ax.add_patch(square)
    
    def init():
        return square,
    
    def update(playhead):
        # Find current index in the values array
        current_idx = np.searchsorted(t, playhead)
        if current_idx > 0:
            # Update square color
            color = colormap(norm(values[current_idx]))
            square.set_facecolor(color)
        return square,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=tqdm(t[::subsample]), 
                        init_func=init, interval=30, blit=True)
    
    # Save the animation with transparent background
    save_path = os.path.join(save_folder, f'color_square_{suffix}.mp4')
    anim.save(save_path, writer='ffmpeg', dpi=300, fps=30,
              savefig_kwargs={'transparent': True, 'facecolor': 'none'})
    plt.close()
    print(colored(f"Saved color square animation to {save_path}", 'green'))



def save_axes(save_folder):
    """Save static images of phase portrait axes"""
    fig, ax = setup_phase_portait_ax(show_ax=True)
    save_path = os.path.join(save_folder, 'phase_portrait_axes.png')
    fig.savefig(save_path)
    plt.close()

def animate_trajectories(t, solutions, colors, save_folder, subsample=1, trail_length=50, suffix="", display_start_times=None, markersize=8, lw=3):
    """
    Animate multiple trajectories in phase space with optional trailing effect
    
    Parameters:
    -----------
    t : array-like
        Time points
    solutions : list of array-like
        List of solution arrays, each with shape (n_points, 2)
    colors : list of str or RGB tuples
        Colors for each trajectory
    save_folder : str
        Folder to save the animation
    subsample : int
        Take every nth point for smoother animation
    trail_length : int or None
        Number of past points to show. If None, show entire past trajectory
    suffix : str
        Suffix for the output filename
    display_start_times : list of float or None
        Times at which to start displaying each trajectory.
        If None, display all trajectories from the beginning.
    """
    fig, ax = setup_phase_portait_ax()
    
    # Convert solutions to points
    points_list = [np.column_stack((sol[:, 0], sol[:, 1])) for sol in solutions]
    
    # Handle display start times

    if display_start_times is None:
        display_start_times = [None] * len(solutions)
    elif not hasattr(display_start_times, '__iter__'):
        display_start_times = [display_start_times] * len(solutions)
    
    if trail_length is None:
        # For full trajectories, create a line for each solution
        lines = []
        current_points = []
        
        for color in colors:
            line = ax.plot([], [], '-', color=color, linewidth=lw, alpha=0.8)[0]
            current_point = ax.plot([], [], 'o', color=color, markersize=markersize)[0]
            lines.append(line)
            current_points.append(current_point)
        
        def init():
            return lines + current_points
        
        def update(playhead):
            # Find current index in the solution array
            current_idx = np.searchsorted(t, playhead)
            
            for i, (points, line, current_point, display_start_time) in enumerate(
                zip(points_list, lines, current_points, display_start_times)):
                
                if display_start_time is not None:
                    # Find index corresponding to display start time
                    start_idx = np.searchsorted(t, display_start_time)
                    
                    # Only show trajectory if we're past the display start time
                    if playhead >= display_start_time:
                        line.set_data(points[start_idx:current_idx+1, 0], 
                                    points[start_idx:current_idx+1, 1])
                    else:
                        line.set_data([], [])
                else:
                    # Show entire trajectory up to current point
                    if current_idx > 0:
                        line.set_data(points[:current_idx+1, 0], points[:current_idx+1, 1])
                
                # Always show current point
                if current_idx > 0:
                    current_point.set_data([points[current_idx, 0]], [points[current_idx, 1]])
            
            return lines + current_points
    
    else:
        # Create line collections for the trailing effect
        line_collections = []
        current_points = []
        
        for color in colors:
            segments = np.zeros((trail_length, 2, 2))
            line_collection = LineCollection(segments, color=color, linewidth=lw)
            # Create alpha array for trailing effect
            alphas = np.linspace(1, 0, trail_length)
            line_collection.set_alpha(alphas)
            ax.add_collection(line_collection)
            line_collections.append(line_collection)
            
            current_point = ax.plot([], [], 'o', color=color, markersize=markersize)[0]
            current_points.append(current_point)
        
        def init():
            return line_collections + current_points
        
        def update(playhead):
            current_idx = np.searchsorted(t, playhead)
            
            for points, line_collection, current_point, display_start_time in zip(
                points_list, line_collections, current_points, display_start_times):
                
                segments = np.zeros((trail_length, 2, 2))
                
                if display_start_time is not None and playhead < display_start_time:
                    # Before display start time, show only current point
                    segments[:] = np.nan
                    line_collection.set_segments(segments)
                else:
                    # Update trail segments
                    for i in range(min(trail_length, current_idx)):
                        idx = current_idx - i
                        if idx > 0:
                            # If using display_start_time, don't show segments before that time
                            if display_start_time is not None:
                                start_idx = np.searchsorted(t, display_start_time)
                                if idx - 1 < start_idx:
                                    segments[i] = np.nan
                                    continue
                            segments[i] = np.column_stack((
                                points[idx-1:idx],
                                points[idx:idx+1]
                            )).reshape(2, 2)
                    
                    # Clear any remaining old segments
                    segments[min(trail_length, current_idx):] = np.nan
                    
                    # Update line collection
                    line_collection.set_segments(segments)
                
                # Always show current point
                if current_idx > 0:
                    current_point.set_data([points[current_idx, 0]], [points[current_idx, 1]])
            
            return line_collections + current_points

    anim = FuncAnimation(fig, update, frames=tqdm(t[::subsample]), 
                        init_func=init, interval=30, blit=True)
    
    save_path = os.path.join(save_folder, f'trajectory_{suffix}.mp4')
    anim.save(save_path, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved trajectory animation to {save_path}", 'green'))


def animate_timeseries(t, values, kind, save_folder, color='white', force_ax_lims=None, lw=3, suffix=''):

    # --- First, save axis to a static figure
    fig, ax = setup_timeseries_ax(kind=kind, show_ax=True)
    ax.set_xlim(t[0], t[-1])
    if force_ax_lims is not None:
        ax.set_ylim(force_ax_lims)
    save_path = os.path.join(save_folder, f'timeseries_{kind}_axes_{suffix}.png')
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(colored(f"Saved {kind} timeseries axes to {save_path}", 'green'))

    # --- Then, animate the time series on the axis without labels
    fig, ax = setup_timeseries_ax(kind=kind, show_ax=False)
    ax.set_xlim(t[0], t[-1])
    if force_ax_lims is not None:
        ax.set_ylim(force_ax_lims)
    
    line = ax.plot([], [], color=color, lw=lw, solid_capstyle='round')[0]
    def init():
        return line,
    def update(frame):
        line.set_data(t[:frame], values[:frame])
        return line,

    anim = FuncAnimation(fig, update, frames=tqdm(range(len(t))), init_func=init, interval=30, blit=True)
    save_path = os.path.join(save_folder, f'timeseries_{kind}_{suffix}.mp4')
    anim.save(save_path, writer='ffmpeg', fps=30, dpi=300)
    plt.close()
    print(colored(f"Saved {kind} timeseries animation to {save_path}", 'green'))


def save_still_trajectory(V, n, color, save_folder, suffix="", lw=3):
    fig, ax = setup_phase_portait_ax()
    ax.plot(V, n, color=color, lw=lw)
    save_path = os.path.join(save_folder, f'trajectory_{suffix}.png')
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(colored(f"Saved still trajectory to {save_path}", 'green'))


def animate_phase_portrait(neuron, currents, save_folder):
    animate_equilibrium_points(neuron, currents, save_folder)
    animate_vector_field(neuron, currents, save_folder)
    animate_streamlines(neuron, currents, save_folder)
    animate_V_nullcline(neuron, currents, save_folder)
    animate_n_nullcline(neuron, currents, save_folder)
    save_axes(save_folder)


