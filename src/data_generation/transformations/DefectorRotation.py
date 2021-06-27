import logging
import numpy as np
import random
from scipy.ndimage.interpolation import rotate
import seaborn as sns
from math import *
from copy import deepcopy
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

def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, cos(theta), -sin(theta)],
                      [0, sin(theta), cos(theta)]])


def Ry(theta):
    return np.matrix([[cos(theta), 0, sin(theta)],
                      [0, 1, 0],
                      [-sin(theta), 0, cos(theta)]])


def Rz(theta):
    return np.matrix([[cos(theta), sin(theta), 0],
                      [sin(theta), cos(theta), 0],
                      [0, 0, 1]])


def rotate_voxels(voxels, angle_x,angle_y,angle_z,axes):
    """
    Rotates the voxels of the model
    :param voxels: Voxels of the model data
    :param angle_x: angle of rotation in axis x
    :param angle_y: angle of rotation in axis y
    :param angle_z: angle of rotation in axis z
    :param axes: a list of axes to consider rotation in (if 0 axis x, 1 axis y, 2 axis z)
    :return: Rotated voxels
    """
    for axis in axes: 
        if axis == 0:
            voxels = voxels.dot(Rx(angle_x))
        elif axis == 1:
            voxels = voxels.dot(Ry(angle_y))
        elif axis == 2:
            voxels = voxels.dot(Rz(angle_z))
    voxels_rotated =  np.asarray(voxels)
    rounded_voxels = np.round(voxels_rotated).astype(int)
    return rounded_voxels, voxels_rotated


def voxel_to_occupancy(voxels):
    """
    Transforms array of voxel indices to occupancy grid
    :param voxels: Voxels of the model data
    :return: Occupancy grid
    """
    '''
    if abs(voxels.max())> abs(voxels.min()):
        N = voxels.max()
    else:
        N = abs(voxels.min())
    '''
    min_x = voxels[:,0].min() 
    min_y =  voxels[:,1].min() 
    min_z = voxels[:,2].min()
    voxels_abs = abs(voxels)
    N = voxels_abs.max()
    #N = voxels.max()
    voxels_occ_grid = np.zeros((N, N, N))
    for coord in voxels:
        # val = coord[0]
        
        x = coord[0]
        y = coord[1]
        z = coord[2]
        if x < 0:
            x = x + 1 
        elif y < 0:
            y = y + 1 
        elif z < 0:
            z = z + 1 
        voxels_occ_grid[x -1][y -1][z -1] = 1

    return voxels_occ_grid


def voxels_in_cylinder(offset, height, radius,voxels):
    #center = np.round(voxels.mean(axis = 0))
    #substruct the center from coordinates
    voxels = voxels
    #voxels inside
    inside_high = voxels[:,2] <= (0.5*height)
    inside_low = voxels[:,2] >= (-0.5*height)
    #projection of x
    inside_circle = np.linalg.norm(voxels[:,:2] - offset ,axis = 1 )<=radius
    return inside_high & inside_low & inside_circle




def rotate_model(model_data: np.ndarray, x_rotation: int, y_rotation: int, z_rotation: int) -> np.ndarray:
    """
    Rotates the voxelized model
    :param model_data: Voxelized model data
    :param x_rotation: Degrees of rotation around the x axis
    :param y_rotation: Degrees of rotation around the y axis
    :param z_rotation:Degrees of rotation around the z axis
    :return: Rotated voxelized model
    """
   
    model_data = np.around(rotate(model_data, x_rotation,(1,2),reshape = False))
    model_data = np.around(rotate(model_data, y_rotation, (0, 2),reshape = False))
    model_data = np.around(rotate(model_data, z_rotation, (0, 1),reshape = False))
    
    return model_data

def rotate_back(model_data: np.ndarray, x_rotation: int, y_rotation: int, z_rotation: int) -> np.ndarray:
    """
    Rotates the voxelized model
    :param model_data: Voxelized model data
    :param x_rotation: Degrees of rotation around the x axis
    :param y_rotation: Degrees of rotation around the y axis
    :param z_rotation:Degrees of rotation around the z axis
    :return: Rotated voxelized model
    """
    model_data = np.around(rotate(model_data, z_rotation, (0, 1),reshape = False))
    model_data = np.around(rotate(model_data, y_rotation, (0, 2),reshape = False))
    model_data = np.around(rotate(model_data, x_rotation,(1,2),reshape = False))
    
    return model_data

def _visualize_top_down_view(model_data: np.ndarray, possible_offsets_final: list):
    """
    Visulizes the top down view of a model
    :param model_data: Voxelized model data
    :param possible_offsets_final: List containing the possible offsets
    :return:
    """
    top_down_view = np.sum(model_data, axis=2)
    #sns.heatmap(basis + (top_down_view > 0))
    sns.heatmap((top_down_view > 0))

    
class DefectorRotation:
    def __init__(self, radius=5, border=5, rotation=False,number_of_trials=5, visualize_top_down_view=False):
        self.radius = radius
        self.border = border
        self.rotation = rotation
        self.visualize_top_down_view = visualize_top_down_view
        self.number_of_trials = number_of_trials
        random.seed(42)

    def __call__(self, model):
        
        model_data = model.model
        #model_data_original = deepcopy(model_data)
        model_data_tmp = deepcopy(model_data)
        #Get the voxels
        voxels = np.argwhere(model_data_tmp == 1)
        
        if voxels.size == 0:
            logging.warning(f"Model empty")
            return None
        
        else: 
            if self.rotation:
                # Rotate model randomly  random.randrange(0, 360)
                x_rotation = random.randrange(0, 360)
                y_rotation = random.randrange(0, 360)
                z_rotation = random.randrange(0, 360)
                
                #pad the model
                padding = int(model_data.shape[0]/2)
                model_data_tmp = np.pad(model_data_tmp, ((padding,padding), (padding,padding), (padding, padding)), 'constant')   
                
                #Rotate the model and preserve the shape
                model_data_tmp = rotate_model(model_data_tmp, x_rotation, y_rotation, z_rotation)
                
                
            #find the offset using model or rotated model
            
            offset, possible_offsets_final = self._find_feasible_offset(model_data_tmp)

            if offset is None:
                logging.warning(f"Could not find a feasable offset for model: {model.model_name}")
                return None
            else: 
                # Define and add the rotated hole
                if self.rotation:
                    # Add a vertical hole in the desired offset
                    model_data_final = add_vertical_hole(model_data_tmp, self.radius, offset)
                    model_data_final = rotate_back(model_data_final,  360- x_rotation, 360- y_rotation, 360- z_rotation)
                    




                else:
                    model_data_final = add_vertical_hole(model_data_tmp, self.radius, offset)

        if self.visualize_top_down_view:
            _visualize_top_down_view(model_data, possible_offsets_final)

        model_with_defect = VoxelModel(model_data_final, np.array([0]), model.model_name + f'_defect_radius{self.radius}')

        return [model, model_with_defect]

    def _find_feasible_offset(self, model_data):
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
            return None, None

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
        try:  # TODO Find problem here
            possible_offsets_final = np.delete(possible_offsets, list(set(to_remove)), axis=0)
        except:
            return None, None

        if len(possible_offsets_final) == 0:
            return None, None

        # Check if the hole has a large enough boarder around it
        for trial in range(self.number_of_trials):
            offset = possible_offsets_final[random.randrange(0, len(possible_offsets_final))]
            offset.astype(int)
            if check_hole_feasibility(model_data, self.radius, self.border, offset):
                break

        if (trial + 1) == self.number_of_trials:
            return None, None

        return offset, possible_offsets_final
