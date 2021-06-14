import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import convolve


def convert_to_hull(model):
    """
    Converts a filled voxel model into a hull of voxels, i.e. removes the voxels inside the model
    :param model: binary np.ndarray of dim 3
    :return: binary np.ndarray of dim 3: Hull of the model
    """
    model_shape = model.shape
    frame_shape = tuple(dim + 6 for dim in model_shape)
    frame = np.zeros(frame_shape)
    frame[3:frame_shape[0] - 3, 3:frame_shape[1] - 3, 3:frame_shape[2] - 3] = model
    kernel = np.ones((3, 3, 3))
    model_out = convolve(model, kernel, mode='constant')
    model_out = np.array(model_out == 27, np.int32)

    return np.array((model_out + model) == 1, np.int32)


def get_layout(title):
    layout = go.Layout(paper_bgcolor='rgb(1,1,1)',
                       title_text=title, title_x=0.5,
                       font_color='white',
                       width=800,
                       height=800,
                       scene_camera=dict(eye=dict(x=1.25, y=-1.25, z=1)),
                       scene_xaxis_visible=False,
                       scene_yaxis_visible=False,
                       scene_zaxis_visible=False)
    return layout


class VoxelModel:
    """
    Model class, for handling voxel models

    :param model: 3 dimensional np.ndarray containing either sdf rep of occupancy grid rep of model
    """
    def __init__(self, model: np.ndarray, label: np.array, model_name: str):
        self.model = model
        self.label = label
        self.model_name = model_name

    def save(self, target_path: str):
        """
        Saves the voxel model as compressed npz array + the label
        :param target_path: target_path where to store the model
        :return: -
        """
        np.savez_compressed(os.path.join(target_path, self.model_name), model=self.model, label=self.label)

    def get_model_data(self):
        """Returns model data (3dim array)"""
        return self.model

    def visualize(self, target_path=None):
        """# TODO"""
        fig = plt.figure(figsize=(6.4*2, 4.8*2))
        ax = fig.gca(projection='3d')
        ax.voxels(self.model)
        ax.set_xlabel('x (0)')
        ax.set_ylabel('y (1)')
        ax.set_zlabel('z (2)')
        if target_path is None:
            plt.show()
        else:
            plt.savefig(os.path.join(target_path, self.model_name))
            plt.close(fig)

    def plot_voxel(self, save=True, format='html'):
        """
        Creates a 3D interactive JavaScript plot of a given 3D voxel model
        :param model: three dimensional numpy.ndarray containing a binary voxel representation
        :param target_name: string containing the name of the output file
        :param save: boolean: Deciding wheter the output model should be saved or not
        :param format: "html", "png", "jpeg": file format of the output file
        :return: Html file containing the plot
        """
        mesh3d_list = []
        model = convert_to_hull(self.model)
        voxel_list = np.transpose(np.nonzero(model))
        for voxel in voxel_list:
            idx = voxel[0]
            idy = voxel[1]
            idz = voxel[2]
            mesh3d = go.Mesh3d(
                # 8 vertices of a cube
                x=[idx, idx, idx + 1, idx + 1, idx, idx, idx + 1, idx + 1],
                y=[idy, idy + 1, idy + 1, idy, idy, idy + 1, idy + 1, idy],
                z=[idz, idz, idz, idz, idz + 1, idz + 1, idz + 1, idz + 1],
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                flatshading=True,
                colorscale=self.colorscale,
                color='#FFFFFF',
                showscale=False
            )
            mesh3d_list.append(mesh3d)
        title = f"STL Model {self.model_name}"
        layout = get_layout(title)
        fig = go.Figure(data=mesh3d_list, layout=layout)
        if save:
            if format == 'html':
                target_path = os.path.join(self.target_dir, self.model_name + '.html')
                fig.write_html(target_path)
            elif format == 'png':
                target_path = os.path.join(self.target_dir, self.model_name + '.png')
                fig.write_image(target_path)
            elif format == 'jpeg':
                target_path = os.path.join(self.target_dir, self.model_name + '.jpeg')
                fig.write_image(target_path)
        else:
            fig.show()
