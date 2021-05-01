import numpy as np
import open3d as o3d

class MeshReader():
    """
    A wrapper class over Open3D's mesh reader function read_triangle_mesh.

    :param path: path to the .obj/.stl file to be read
    """
    def __init__(self, path):
        self.path = path
        self.mesh = o3d.io.read_triangle_mesh(path)

    def visualize(self, geometries):
        """
        A wrapper over Open3D's visualization function draw_geometries 
        which takes a list of geometry objects and renders them together.

        It views the loaded mesh and the given geometries together.

        :param geometries: a list o3d.geometry objects
        """

        o3d.visualization.draw_geometries([self.mesh] + geometries)
