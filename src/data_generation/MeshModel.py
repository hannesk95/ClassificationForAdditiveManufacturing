import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from src.data_generation.utils import extract_file_name


class MeshModel:
    """
    Model class, for handling mesh models

    :param path: Path to the stl file
    """
    def __init__(self, path):
        self.path = path
        self.mesh, self.vertices, self.normals, self.faces = self._load_model()
        self.model_name = extract_file_name(path)

    def _load_model(self):
        mesh = o3d.io.read_triangle_mesh(self.path)
        mesh.compute_vertex_normals()
        return mesh, np.asarray(mesh.vertices), mesh.triangle_normals, np.asarray(mesh.triangles)

    def save(self, target_path=None):
        if target_path is None:
            target_path = self.path

        # from numpy array to o3d Vector3dVector
        vertices_o3d = o3d.utility.Vector3dVector(self.vertices)
        # from numpy array to o3d Vector3iVector
        faces_o3d = o3d.utility.Vector3iVector(self.faces)
        # create o3d TriangleMesh
        mesh = o3d.geometry.TriangleMesh(vertices=vertices_o3d, triangles=faces_o3d)
        # compute normal to save as stl file
        mesh.compute_triangle_normals()
        # save the mesh to path
        o3d.io.write_triangle_mesh(target_path, mesh)

    def get_model_data(self):
        """Returns vertices, normals and faces"""
        return self.vertices, self.normals, self.faces

    def set_model_data(self, vertices, normals, faces):
        self.vertices = vertices
        self.normals = normals
        self.faces = faces

    def visualize(self, geometries):
        """
        A wrapper over Open3D's visualization function draw_geometries 
        which takes a list of geometry objects and renders them together.
        It views the loaded mesh and the given geometries together.
        :param geometries: a list o3d.geometry objects
        """

        o3d.visualization.draw_geometries([self.mesh] + geometries)
