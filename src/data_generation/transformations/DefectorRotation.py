import numpy as np
import random
from scipy.ndimage.interpolation import rotate
import seaborn as sns

from src.data_generation.VoxelModel import VoxelModel


def add_vertical_hole(model_data, radius, offset):
    xx = np.arange(model_data.shape[0])
    yy = np.arange(model_data.shape[1])
    out = np.zeros_like(model_data)

    for idz in range(model_data.shape[2]):
        voxel_sim_defect_circle_cut = model_data[:, :, idz].astype(np.int32)
        inside = (xx[:, None] - offset[0]) ** 2 + (yy[None, :] - offset[1]) ** 2 > (radius ** 2)
        out[:, :, idz] = (voxel_sim_defect_circle_cut & inside)

    model_data_with_defect = np.array(out)

    return model_data_with_defect


def determine_first_unique_horizontal_elements(source, number_of_elements):
    to_remove = []
    value = 0
    idx = 0
    while idx < len(source):
        offset = source[idx]
        if offset != value:
            for idy in range(number_of_elements):
                if idx + idy >= len(source):
                    break
                to_remove.append(idx + idy)
            idx += number_of_elements
            value = offset
        idx += 1

    return to_remove


def determine_last_unique_horizontal_elements(source, number_of_elements):
    to_remove = []
    idx = 0
    while idx < len(source) - number_of_elements:
        value = source[idx]
        offset = source[idx + number_of_elements]
        if offset != value:
            for idy in range(number_of_elements):
                if idx + idy > len(source):
                    break
                to_remove.append(idx + idy)
            idx += number_of_elements
            value = offset
        idx += 1

    for idy in range(number_of_elements):
        to_remove.append(len(source) - idy - 1)

    return to_remove


class DefectorRotation:
    def __init(self, radius=2, border=5, rotation=True, visualize_top_down_view=False):
        self.radius = radius
        self.border = border
        self.rotation = rotation
        self.visualize_top_down_view = visualize_top_down_view

    def __call__(self, model):
        model_data = model.model

        if self.rotation:
            # Rotate model randomly
            x_rotation = random.randrange(0, 360)
            y_rotation = random.randrange(0, 360)
            z_rotation = random.randrange(0, 360)
            model_data = np.around(rotate(model_data, x_rotation))
            model_data = np.around(rotate(model_data, y_rotation, (1, 2)))
            model_data = np.around(rotate(model_data, z_rotation, (0, 2)))

        # Get top down view and all non-zero elements in the top down view
        top_down_view = np.sum(model_data, axis=2)
        possible_offsets = np.array(np.where(top_down_view > 0)).T

        to_remove = []
        # Define horizontal elements to be removed
        to_remove += determine_first_unique_horizontal_elements(possible_offsets[:, 0], self.border)
        to_remove += determine_last_unique_horizontal_elements(possible_offsets[:, 0], self.border)

        # Define vertical elements to be removed
        for value in list(set(possible_offsets[:, 1])):
            values = np.where(possible_offsets[:, 1] == value)[0]
            to_remove += values[:self.border].tolist()
            to_remove += values[len(values) - self.border:len(values)].tolist()

        # Remove elements at the border
        possible_offsets_final = np.delete(possible_offsets, list(set(to_remove)), axis=0)

        offset = possible_offsets_final[random.randrange(0, len(possible_offsets_final))]
        offset.astype(int)
        model_data = add_vertical_hole(model.model, self.radius, offset)

        if self.rotation:
            # Rotate model back
            model_data = np.around(rotate(model_data, 360-x_rotation))
            model_data = np.around(rotate(model_data, 360-y_rotation, (1, 2)))
            model_data = np.around(rotate(model_data, 360-z_rotation, (0, 2)))

        # TODO Put model in shape as before
        # TODO Add more checks
        # TODO Find way on how to save model with defect (first idea return list of models
        
        if self.visualize_top_down_view:
            top_down_view = np.sum(model_data, axis=2)
            sns.heatmap(top_down_view)

        model_with_defect = VoxelModel(model_data, np.array([0]), model.model_name + f'_defect_radius{self.radius}')

        return model
