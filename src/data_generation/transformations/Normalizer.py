import numpy as np
import open3d as o3d
from stl import mesh
import pymeshlab

class Normalizer:
    """# TODO"""
    def __call__(self, model):
        """# TODO"""
        mesh = model.mesh
        self.center_mesh_around_origin(mesh)
        scale_factor = 1 / np.max(mesh.get_max_bound() - mesh.get_min_bound())
        self.scale(mesh, scale_factor)
        model.set_model_data(np.asarray(mesh.vertices), np.asarray(mesh.triangle_normals),
                                 np.asarray(mesh.triangles))
        return model

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
