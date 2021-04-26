import numpy as np
import re
import os.path
from stl import mesh
import plotly.graph_objects as go


def plot_mesh(stl_path, target_name=None, target_dir='data/plots'):
    """
    Creates a 3D interactive JavaScript plot of a given stl 3D model

    :param stl_path: Source path to the stl model to plot
    :param target_name: name of the resulting html file
    :param target_dir: path to the target path where to save the created html file
    :return: Creates a html file containing the plot
    """
    if not os.path.isfile(stl_path):
        raise TypeError(f'No file found at {stl_path}. Provide a valid path ')

    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', target_dir))
    if not os.path.isdir(target_dir):
        raise TypeError(f'No directory found at {target_dir}. Provide a valid directory path ')

    if target_name is None:
        target_name = stl_path.split('/')[-1]
        target_name = re.sub('.stl$', '', target_name)

    my_mesh = mesh.Mesh.from_file(stl_path)

    p, q, r = my_mesh.vectors.shape  # (p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(my_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3 * k for k in range(p)])
    J = np.take(ixr, [3 * k + 1 for k in range(p)])
    K = np.take(ixr, [3 * k + 2 for k in range(p)])

    x, y, z = vertices.T
    colorscale = [[0, '#e5dee5'], [1, '#e5dee5']]

    mesh3D = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=I,
        j=J,
        k=K,
        flatshading=True,
        colorscale=colorscale,
        intensity=z,
        name=target_name,
        showscale=False)

    title = f"STL Model {target_name}"
    layout = go.Layout(paper_bgcolor='rgb(1,1,1)',
                       title_text=title, title_x=0.5,
                       font_color='white',
                       width=800,
                       height=800,
                       scene_camera=dict(eye=dict(x=1.25, y=-1.25, z=1)),
                       scene_xaxis_visible=False,
                       scene_yaxis_visible=False,
                       scene_zaxis_visible=False)
    fig = go.Figure(data=[mesh3D], layout=layout)

    target_path = os.path.join(target_dir, target_name + '.html')
    fig.write_html(target_path)
