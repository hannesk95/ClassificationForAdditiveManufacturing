from mesh_to_sdf import mesh_to_voxels
import numpy as np
import trimesh
from skimage import measure
from src.data_generation.VoxelModel import VoxelModel


def sdf_to_binary(sdf):
    """
    The function takes in sdf values and converts it into a binary field.
    : sdf: N x N x N array of sdf values where N is the number of voxels.

    """
    binay_voxels = np.ones_like(sdf)
    dims = [sdf.shape[0], sdf.shape[1], sdf.shape[2]]

    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if abs(float(sdf[i][j][k])) > 0.05:
                    binay_voxels[i][j][k] = 0
    return binay_voxels


def visualise_sdf(sdf):
    """
    The function reconstructs the mesh using marching cubes algorithm and renders it.
    : sdf: N x N x N array of sdf values where N is the number of voxels.
    """
    vertices, faces, normals, _ = measure.marching_cubes(sdf, level=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh.show()


def sdf_to_tsdf(sdf):
    """
    The function takes in sdf values and converts it into TSDF values.Rasterization is then done using Marching Cubes algorithm.
    : sdf: N x N x N array of sdf values where N is the number of voxels.
    """

    dims = [sdf.shape[0], sdf.shape[1], sdf.shape[2]]

    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if float(sdf[i][j][k]) > 1:
                    sdf[i][j][k] = 1.0
                elif float(sdf[i][j][k]) < -0.5:
                    sdf[i][j][k] = -0.5
    return sdf


class Voxelizer:
    def __init__(self, dimension=64, representation='occupancy'):
        self.dimension = dimension
        self.representation = representation

    def __call__(self, model):
        voxels = self.mesh_to_sdf(model)
        if self.representation == 'sdf':
            model = VoxelModel(voxels, np.array([1]), model.model_name)
        else:
            voxels = sdf_to_binary(voxels)
            model = VoxelModel(voxels, np.array([1]), model.model_name)
        return model

    def mesh_to_sdf(self, model):
        """
        The function converts an input triangular mesh into a voxelized array.
        """
        vertices, normals, faces = model.get_model_data()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        voxels = mesh_to_voxels(mesh, self.dimension, sign_method='depth', pad=False)
        return voxels
