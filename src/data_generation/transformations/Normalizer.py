import numpy as np
import open3d as o3d
from stl import mesh
import pymeshlab

class Normalizer:
    def __init__(self, mesh):
        # TODO: Add the parameters to init
        self.mesh = mesh  # TODO: Find common stl datatyp

    def __call__(self):
        self.center_mesh_around_origin(self.mesh.mesh)
        scale_factor = 1 / np.max(self.mesh.mesh.get_max_bound() - self.mesh.mesh.get_min_bound())
        self.scale(self.mesh.mesh, scale_factor)
        self.mesh.set_model_data(np.asarray(self.mesh.mesh.vertices), np.asarray(self.mesh.mesh.triangle_normals), np.asarray(self.mesh.mesh.triangles))

    def center_mesh_around_origin(self, mesh):
        """
        Translates the mesh to the origin in-place

        :param mesh: O3D mesh object
        """
        centered_vertices = np.array(mesh.vertices) - mesh.get_center()
        mesh.vertices = o3d.utility.Vector3dVector(centered_vertices)

    def scale(self, mesh, scale_factor):
        """
        A wrapper over Open3D's scaling function scale.

        It scales the mesh object in-place.

        :param scale_factor: factor by which the mesh should be scaled
        """
        mesh.scale(scale_factor, center=mesh.get_center())
