import os
import numpy as np
import matplotlib.pyplot as plt


class VoxelModel:
    """
    Model class, for handling voxel models

    :param model: 3 dimensional np.ndarray containing either sdf rep of occupancy grid rep of model
    """
    def __init__(self, model: np.array, label: np.array, model_name: str):
        self.model = model
        self.label = label
        self.model_name = model_name

    def save(self, target_path: str):
        """
        Saves the voxel model as compressed npz array + the label
        :param target_path: target_path where to store the model
        :return: -
        """
        np.savez_compressed(os.path.join(target_path, self.model_name), model=self.voxel_rep, label=self.label)

    def get_model_data(self):
        """Returns model data (3dim array + label)"""
        return self.vertices, self.normals, self.faces

    def visualize(self, target_path=None):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(self.model)
        if target_path is None:
            plt.show()
        else:
            plt.savefig(os.path.join(target_path, self.model_name))
