import numpy as np
import open3d as o3d

class MeshSaver():
    """
    Save mesh into specified path from numpy vertices and faces.

    :param path: path where to save the .stl file
    :param vertices: numpy array containing vertices
    :param faces: numpy array containing faces

    """
    def __init__(self, path,vertices,faces):
        self.path = path
        self.vertices = vertices
        self.faces = faces
    def save(self):
        #from numpy array to o3d Vector3dVector
        vertices_o3d = o3d.utility.Vector3dVector(self.vertices)
        # from numpy array to o3d Vector3iVector
        faces_o3d = o3d.utility.Vector3iVector(self.faces)
        #create o3d TriangleMesh
        mesh = o3d.geometry.TriangleMesh(vertices=vertices_o3d, triangles=faces_o3d)
        #compute normal to save as stl file
        mesh.compute_triangle_normals()
        #save the mesh to path
        o3d.io.write_triangle_mesh(self.path, mesh)

