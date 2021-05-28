import logging
import numpy as np
import random
from scipy.ndimage.interpolation import rotate
import seaborn as sns

from src.data_generation.VoxelModel import VoxelModel


def add_vertical_hole(model_data, radius, offset):
    xx = np.arange(model_data.shape[0])
    yy = np.arange(model_data.shape[1])
    out = np.zeros_like(model_data)
    inside = (xx[:, None] - offset[0]) ** 2 + (yy[None, :] - offset[1]) ** 2 > (radius ** 2)

    for idz in range(model_data.shape[2]):
        voxel_sim_defect_circle_cut = model_data[:, :, idz].astype(np.int32)
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


def check_hole_feasibility(model_data, radius, offset):
    xx = np.arange(model_data.shape[0])
    yy = np.arange(model_data.shape[1])
    top_down_view = np.sum(model_data, axis=2)
    inside = (xx[:, None] - offset[0]) ** 2 + (yy[None, :] - offset[1]) ** 2 > (radius ** 2)
    inidices_to_remove = np.array(np.where(inside is False)).T
    for indices in inidices_to_remove:
        if top_down_view[indices[0], indices[1]] == 0:
            return False
    return True


class DefectorRotation:
    def __init__(self, radius=2, border=5, rotation=True, visualize_top_down_view=False, number_of_trials=5):
        self.radius = radius
        self.border = border
        self.rotation = rotation
        self.visualize_top_down_view = visualize_top_down_view
        self.number_of_trials = number_of_trials

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
        if len(possible_offsets) == 0:
            return model # TODO return empty list

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
        if len(possible_offsets_final) == 0:
            return model # TODO return empty list

        for trial in range(self.number_of_trials):
            offset = possible_offsets_final[random.randrange(0, len(possible_offsets_final))]
            offset.astype(int)
            if check_hole_feasibility(model_data, self.radius, offset):
                break

        if (trial-1) == self.number_of_trials:
            logging.warning(f'Could not find feasible offset for model: {model.model_name}')
            # TODO Think about returning empty list
            return model

        model_data = add_vertical_hole(model.model, self.radius, offset)

        if self.rotation:
            # Rotate model back
            model_data = np.around(rotate(model_data, 360 - x_rotation))
            model_data = np.around(rotate(model_data, 360 - y_rotation, (1, 2)))
            model_data = np.around(rotate(model_data, 360 - z_rotation, (0, 2)))

        # TODO Put model in shape as before

        if self.visualize_top_down_view:
            basis = np.zeros_like(top_down_view)
            for indices in possible_offsets_final:
                idx = indices[0]
                idy = indices[1]
                basis[idx, idy] = 1
            top_down_view = np.sum(model_data, axis=2)
            sns.heatmap(basis + (top_down_view > 0))

        model_with_defect = VoxelModel(model_data, np.array([0]), model.model_name + f'_defect_radius{self.radius}')

        return [model, model_with_defect]
