import logging
import numpy as np
import random
from scipy.ndimage.interpolation import rotate
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_generation.VoxelModel import VoxelModel


def add_vertical_hole(model_data: np.ndarray, radius: int, offset: np.ndarray) -> np.ndarray:
    """
    Adds a hole through the z axis
    :param model_data: Voxelized model data
    :param radius: Radius of the hole
    :param offset: Offset, i.e. where to put the hole
    :return: model_data_with_defect: voxelized model with a added hole
    """
    xx = np.arange(model_data.shape[0])
    yy = np.arange(model_data.shape[1])
    out = np.zeros_like(model_data)
    inside = (xx[:, None] - offset[0]) ** 2 + (yy[None, :] - offset[1]) ** 2 > (radius ** 2)

    for idz in range(model_data.shape[2]):
        voxel_sim_defect_circle_cut = model_data[:, :, idz].astype(np.int32)
        out[:, :, idz] = (voxel_sim_defect_circle_cut & inside)

    model_data_with_defect = np.array(out)

    return model_data_with_defect


def determine_first_unique_horizontal_elements(source: list, number_of_elements: int) -> list:
    """
    Determines the first number_of_elements from the left in order to remove them
    :param source: List of indices
    :param number_of_elements: Number of indices to be removed from the left
    :return: List containing the indices to be removed
    """
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


def determine_last_unique_horizontal_elements(source: list, number_of_elements: int) -> list:
    """
    Determines the first number_of_elements from the right in order to remove them
    :param source: List of indices
    :param number_of_elements: Number of indices to be removed from the right
    :return: List containing the indices to be removed
    """
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


def check_hole_feasibility(model_data: np.ndarray, radius: int, border: int, offset: np.ndarray) -> bool:
    """
    Checks whether a larger hole (radius+offset) around the offset would be still fully in the model
    :param model_data: Voxelized model data
    :param radius: Radius of the hole to be added
    :param border: Border in each direction
    :param offset: Offset parameter
    :return: Boolean indicating whether or nor a hole with radius radius+offset at offset would be fully in the model
    """
    radius += border
    top_down_view = np.sum(model_data, axis=2)
    top_down_view = np.pad(top_down_view, pad_width=1, mode='constant', constant_values=0)
    xx = np.arange(top_down_view.shape[0])
    yy = np.arange(top_down_view.shape[1])
    inside = (xx[:, None] - offset[0]) ** 2 + (yy[None, :] - offset[1]) ** 2 > (radius ** 2)
    inidices_to_remove = np.array(np.where(inside == False)).T
    for indices in inidices_to_remove:
        if top_down_view[indices[0], indices[1]] == 0:
            return False
    return True


def rotate_model(model_data: np.ndarray, x_rotation: int, y_rotation: int, z_rotation: int) -> np.ndarray:
    """
    Rotates the voxelized model
    :param model_data: Voxelized model data
    :param x_rotation: Degrees of rotation around the x axis
    :param y_rotation: Degrees of rotation around the y axis
    :param z_rotation:Degrees of rotation around the z axis
    :return: Rotated voxelized model
    """
    model_data = np.around(rotate(model_data, x_rotation))
    model_data = np.around(rotate(model_data, y_rotation, (1, 2)))
    model_data = np.around(rotate(model_data, z_rotation, (0, 2)))

    return model_data


def _visualize_top_down_view(model_data: np.ndarray, possible_offsets_final: list):
    """
    Visulizes the top down view of a model
    :param model_data: Voxelized model data
    :param possible_offsets_final: List containing the possible offsets
    :return:
    """
    top_down_view = np.sum(model_data, axis=2)
    basis = np.zeros_like(top_down_view)
    for indices in possible_offsets_final:
        idx = indices[0]
        idy = indices[1]
        basis[idx, idy] = 1
    sns.heatmap(basis + (top_down_view > 0))
    plt.show()


class DefectorRotation:
    def __init__(self, hole_radius_nonprintable: int = 5, hole_radius_printable: int = 10, border_nonprintable: int = 3,
                 border_printable: int = 5, rotation: bool = False, number_of_trials: int = 5,
                 visualize_top_down_view: bool = False):

        self.hole_radius_nonprintable = hole_radius_nonprintable
        self.hole_radius_printable = hole_radius_printable

        self.border_nonprintable = border_nonprintable
        self.border_printable = border_printable

        self.rotation = rotation
        self.visualize_top_down_view = visualize_top_down_view
        self.number_of_trials = number_of_trials
        random.seed(42)

    def __call__(self, model):
        model_data = model.model

        if self.rotation:
            # Rotate model randomly
            x_rotation = random.randrange(0, 360)
            y_rotation = random.randrange(0, 360)
            z_rotation = random.randrange(0, 360)
            model_data = rotate_model(model_data, x_rotation, y_rotation, z_rotation)

        model_data_nonprintable_defect_middle = self._add_defect_middle(model_data, self.hole_radius_nonprintable,
                                                                        self.border_nonprintable)
        model_data_printable_defect_middle = self._add_defect_middle(model_data, self.hole_radius_printable,
                                                                     self.border_printable)
        model_data_nonprintable_defect_border = self._add_defect_border(model_data, self.hole_radius_printable,
                                                                        self.border_nonprintable)

        if self.rotation:
            # Rotate model back
            model_data = rotate_model(model_data, 360-x_rotation, 360-y_rotation, 360-z_rotation)

        out = []
        if model_data_nonprintable_defect_middle is not None:
            out.append(model)
            model_nonprintable_defect_middle = VoxelModel(model_data_nonprintable_defect_middle, np.array([0]),
                                                          model.model_name +
                                                          f'_nonprintable_defect_middle{self.hole_radius_nonprintable}')
            out.append(model_nonprintable_defect_middle)

        if model_data_printable_defect_middle is not None and model_data_nonprintable_defect_border is not None:
            model_printable_defect_middle = VoxelModel(model_data_printable_defect_middle, np.array([1]),
                                                       model.model_name +
                                                       f'_printable_defect_middle{self.hole_radius_printable}')
            model_nonprintable_defect_border = VoxelModel(model_data_nonprintable_defect_border, np.array([0]),
                                                          model.model_name +
                                                          f'_nonprintable_defect_border{self.border_nonprintable}')
            out.append(model_printable_defect_middle)
            out.append(model_nonprintable_defect_border)

        return out

    def _add_defect_middle(self, model_data, radius, border):
        offset = self._find_feasible_offset_middle(model_data, radius, border)
        if offset is None:
            # logging.warning(f"Could not find a feasable offset for model: {model.model_name}")
            return None
        model_data = add_vertical_hole(model_data, radius, offset)

        if self.visualize_top_down_view:
            _visualize_top_down_view(model_data, [])

        return model_data

    def _find_feasible_offset_middle(self, model_data, radius, border):
        """
        Finds a feasible offset where to put randomly hole. It analyses the top down view of a model and samples a
        offset from a adapted subsample of offsets, that guarantee that the hole fully goes through the model
        :param model_data:
        :return: offset (np.ndarray), possible_offsets_final (np.ndarray, containing the final subset of offsets
        """
        # Get top down view and all non-zero elements in the top down view
        top_down_view = np.sum(model_data, axis=2)
        possible_offsets = np.array(np.where(top_down_view > 0)).T

        if len(possible_offsets) == 0:
            return None

        to_remove = self._determine_indices_to_remove(possible_offsets, border)

        # Remove elements at the border
        try:  # TODO Find problem here
            possible_offsets_final = np.delete(possible_offsets, to_remove, axis=0)
        except:
            return None

        if len(possible_offsets_final) == 0:
            return None

        # Check if the hole has a large enough boarder around it
        for trial in range(self.number_of_trials):
            offset = possible_offsets_final[random.randrange(0, len(possible_offsets_final))]
            offset.astype(int)
            if check_hole_feasibility(model_data, radius, border, offset):
                break

        if (trial + 1) == self.number_of_trials:
            return None

        if self.visualize_top_down_view:
            _visualize_top_down_view(model_data, possible_offsets_final)

        return offset

    def _add_defect_border(self, model_data, radius, border):
        offset = self._find_feasible_offset_border(model_data, radius, border)
        if offset is None:
            # logging.warning(f"Could not find a feasable offset for model: {model.model_name}")
            return None
        model_data = add_vertical_hole(model_data, radius, offset)

        if self.visualize_top_down_view:
            _visualize_top_down_view(model_data, [])

        return model_data

    def _determine_indices_to_remove(self, possible_offsets, border):
        to_remove = []
        # Define horizontal elements to be removed
        to_remove += determine_first_unique_horizontal_elements(possible_offsets[:, 0], border)
        to_remove += determine_last_unique_horizontal_elements(possible_offsets[:, 0], border)
        # Define vertical elements to be removed
        for value in list(set(possible_offsets[:, 1])):
            values = np.where(possible_offsets[:, 1] == value)[0]
            to_remove += values[:border].tolist()
            to_remove += values[len(values) - border:len(values)].tolist()

        return list(set(to_remove))

    def _find_feasible_offset_border(self, model_data, radius, border):
        top_down_view = np.sum(model_data, axis=2)
        possible_offsets = np.array(np.where(top_down_view > 0)).T

        to_remove = self._determine_indices_to_remove(possible_offsets, border+radius+1)

        try:  # TODO Find problem here
            possible_offsets_final = possible_offsets[list(set(to_remove))]
        except:
            return None

        to_remove = self._determine_indices_to_remove(possible_offsets_final, radius)

        if len(possible_offsets_final) == 0:
            return None

        try:  # TODO Find problem here
            possible_offsets_final = np.delete(possible_offsets_final, to_remove, axis=0)
        except:
            return None

        if len(possible_offsets_final) == 0:
            return None

        for trial in range(self.number_of_trials):
            offset = possible_offsets_final[random.randrange(0, len(possible_offsets_final))]
            offset.astype(int)
            if check_hole_feasibility(model_data, radius, 1, offset):
                break

        if (trial + 1) == self.number_of_trials:
            return None

        if self.visualize_top_down_view:
            _visualize_top_down_view(model_data, possible_offsets_final)

        return offset
