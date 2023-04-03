import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.animation

from tqdm import tqdm_notebook as tqdm
import cmasher

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map

        Taken from https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
        '''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def make_segments(x,y):
    '''
        Create segments for LineCollection
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)

def get_black_patch(t_pos, ax, color="black"):
    '''
        Utility to animate playing effect.
        Returns a black patch, masking axis region, starting from t_pos
    '''
    rect_x=t_pos
    rect_y= ax.get_ylim()[0]
    rect_w = ax.get_xlim()[1]- rect_x
    rect_h = ax.get_ylim()[1]-rect_y
    return Rectangle((rect_x, rect_y),rect_w, rect_h,facecolor=color)

def setup_black_axes(figsize=(12,5), dpi=300):
    fig, ax = plt.subplots(1,1,figsize=figsize,dpi=dpi)
    ax.set_facecolor("black")
    fig.set_facecolor("black")
    ax.axis(False)
    return fig,ax

def animate_lines(lines, fig, ANIMATION_FRAMES=None):
    
    x_datas = [line.get_xdata() for line in lines]
    y_datas = [line.get_ydata() for line in lines]
    if ANIMATION_FRAMES is None:
        ANIMATION_FRAMES = x_datas[0]
    
    def anim_func(playhead_t):
        for k,line in enumerate(lines):
            mask = x_datas[k]<playhead_t
            line.set_xdata(x_datas[k][mask])
            line.set_ydata(y_datas[k][mask])
        return lines,
    
    return matplotlib.animation.FuncAnimation(fig, anim_func,frames=tqdm(ANIMATION_FRAMES),interval=30)
    
def get_gradient_curve(x,y,cmap=cmasher.guppy_r, color_by="x", linewidths=10, **kwargs):
    '''
        Creates an instance of LineCollection representing a curve (x,y) colored by x or y according to a cmap.
    '''
    lc = matplotlib.collections.LineCollection(make_segments(x, y),linewidths=linewidths, **kwargs)
    if color_by=="x":   
        lc.set_array(x/np.max(x))
    if color_by=="y":
        lc.set_array(y/np.max(y))
        
    lc.set_capstyle("round")
    lc.set_cmap(cmap)
    return lc


def animate_line_collections(line_collections, fig, ANIMATION_FRAMES=None):

    x_datas = []
    y_datas = []
       
    for lc in line_collections:
        segments = lc.get_segments()
        x_datas.append(np.array([seg[0,0] for seg in segments]))
        y_datas.append(np.array([seg[0,1] for seg in segments]))
        
    if ANIMATION_FRAMES is None:
        ANIMATION_FRAMES = x_datas[0]
        
    def anim_func(playhead_t):
        for k,lc in enumerate(line_collections):
            mask = x_datas[k]<playhead_t
            lc.set_segments(make_segments(x_datas[k][mask], y_datas[k][mask]))
            
        return line_collections,

    return matplotlib.animation.FuncAnimation(fig, anim_func,frames=tqdm(ANIMATION_FRAMES),interval=30)


def get_animated_fill(x,y, fig,  ax, ANIMATION_FRAMES=None,color="white",**kwargs):
    
    fill =ax.fill_between(x,0,y,color=color,**kwargs)
    
    if ANIMATION_FRAMES is None:
        ANIMATION_FRAMES = x
        
    def anim_func(playhead_x):
        mask = x>playhead_x
        ax.collections.clear()
        x_masked=np.ma.masked_where(mask,x)
        y_masked=np.ma.masked_where(mask,y)
        fill = ax.fill_between(x_masked,0,y_masked,color=color, **kwargs)
        return fill,

    return (fill, matplotlib.animation.FuncAnimation(fig, anim_func,frames=tqdm(ANIMATION_FRAMES),interval=30))
    