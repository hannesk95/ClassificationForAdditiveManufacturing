import os
import numpy as np
import matplotlib.pyplot as plt


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
