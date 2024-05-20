#############################################
#                                           #
# code from IRVAE_public (Lee et al., 2022) #
#                                           #
#############################################

import yaml
import os
import numpy as np
from tqdm.auto import tqdm, trange
from matplotlib.patches import Ellipse, Rectangle, Polygon
import requests as req
import shutil

import io 
from PIL import Image

def download_file(path_from, path_to, pbar=True):
    r = req.get(path_from, stream=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    
    file_size = int(r.headers.get('Content-Length', 0))
    
    os.makedirs(os.path.dirname(path_to), exist_ok=True)
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=os.path.basename(path_from), disable=not pbar, ncols=150) as r_raw:
        with open(path_to, 'wb') as f:
            shutil.copyfileobj(r_raw, f)

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)

def label_to_color(label):
    
    n_points = label.shape[0]
    color = np.zeros((n_points, 3))

    # color template (2021 pantone color: orbital)
    rgb = np.zeros((11, 3))
    rgb[0, :] = [253, 134, 18]
    rgb[1, :] = [106, 194, 217]
    rgb[2, :] = [111, 146, 110]
    rgb[3, :] = [153, 0, 17]
    rgb[4, :] = [179, 173, 151]
    rgb[5, :] = [245, 228, 0]
    rgb[6, :] = [255, 0, 0]
    rgb[7, :] = [0, 255, 0]
    rgb[8, :] = [0, 0, 255]
    rgb[9, :] = [18, 134, 253]
    rgb[10, :] = [155, 155, 155] # grey

    for idx_color in range(10):
        color[label == idx_color, :] = rgb[idx_color, :]
    return color


def attr_to_color(label):
    
    n_points = label.shape[0]
    color = np.zeros((n_points, 3))

    # color template (2021 pantone color: orbital)
    rgb = np.zeros((3, 3))
    rgb[0, :] = [253, 134, 18]      # orange for female
    rgb[1, :] = [106, 194, 217]     # blue for male
    rgb[2, :] = [155, 155, 155]    # grey

    for idx_color in range(2):
        color[label == idx_color, :] = rgb[idx_color, :]
    return color

def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def PD_metric_to_ellipse(G, center, scale, **kwargs):
    
    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(G)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # find angle of ellipse
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # draw ellipse
    if (eigvals > 0).all():
        width, height = 2 * scale * np.sqrt(eigvals)
    else:
        width, height = 0, 0
    return Ellipse(xy=center, width=width, height=height, angle=np.degrees(theta), **kwargs)

def rectangle_scatter(size, center, color):

    return Rectangle(xy=(center[0]-size[0]/2, center[1]-size[1]/2) ,width=size[0], height=size[1], facecolor=color)

def triangle_scatter(size, center, color):
    
    return Polygon(((center[0], center[1] + size[1]/2), (center[0] - size[0]/2, center[1] - size[1]/2), (center[0] + size[0]/2, center[1] - size[1]/2)), fc=color)

class progress_tracker():
    def __init__(self, **kwargs):
        self.tqdm_obj = tqdm(**kwargs)
        self.value_old = 0
        
    def update(self, x):
        self.tqdm_obj.update(x - self.value_old)
        self.value_old = x
        self.tqdm_obj.refresh()
    
    def close(self):
        self.tqdm_obj.close()
        
def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)