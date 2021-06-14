import numpy as np
import os
from stl import mesh
import open3d as o3d
import plotly.graph_objects as go
from src.data_generation.utils import extract_file_name
from src.data_generation.VoxelModel import get_layout


def _convert_to_o3d_mesh(vertices, faces):
    """
    # TODO
    :param vertices:
    :param faces:
    :return:
    """
    # from numpy array to o3d Vector3dVector
    vertices_o3d = o3d.utility.Vector3dVector(vertices)
    # from numpy array to o3d Vector3iVector
    faces_o3d = o3d.utility.Vector3iVector(faces)
    # create o3d TriangleMesh
    mesh = o3d.geometry.TriangleMesh(vertices=vertices_o3d, triangles=faces_o3d)
    # compute normal to save as stl file
    mesh.compute_triangle_normals()

    return mesh


class MeshModel:
    """
    Model class, for handling mesh models
    # TODO Add docstrings
    :param path: Path to the stl file
    """
    def __init__(self, path):
        """# TODO"""
        self.path = path
        self.mesh, self.vertices, self.normals, self.faces = self._load_model()
        self.model_name = extract_file_name(path)

    def _load_model(self):
        """# TODO"""
        mesh = o3d.io.read_triangle_mesh(self.path)
        mesh.compute_vertex_normals()
        return mesh, np.asarray(mesh.vertices), mesh.triangle_normals, np.asarray(mesh.triangles)

    def save(self, target_path=None):
        """# TODO"""
        if target_path is None:
            target_path = self.path

        mesh = _convert_to_o3d_mesh(self.vertices, self.faces)
        # save the mesh to path
        o3d.io.write_triangle_mesh(target_path, mesh)

    def get_model_data(self):
        """Returns vertices, normals and faces"""
        return self.vertices, self.normals, self.faces

    def set_model_data(self, vertices, normals, faces):
        """# TODO"""
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
        self.mesh = _convert_to_o3d_mesh(vertices, faces)

    def visualize(self, geometries):
        """
        A wrapper over Open3D's visualization function draw_geometries 
        which takes a list of geometry objects and renders them together.
        It views the loaded mesh and the given geometries together.
        :param geometries: a list o3d.geometry objects
        """

        o3d.visualization.draw_geometries([self.mesh] + geometries)

    def plot_mesh(self, target_name=None, save=True):
        """
        Creates a 3D interactive JavaScript plot of a given stl 3D model

        :param stl_path: Source path to the stl model to plot
        :param target_name: name of the resulting html file. If not provided the name of the given file in stl_path will be
        used
        :param save: boolean: Deciding wheter the output model should be saved or not
        :return: Html file containing the plot
        """
        my_mesh = mesh.Mesh.from_file(self.path)

        p, q, r = my_mesh.vectors.shape  # (p, 3, 3)
        # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
        # extract unique vertices from all mesh triangles
        vertices, ixr = np.unique(my_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0)
        I = np.take(ixr, [3 * k for k in range(p)])
        J = np.take(ixr, [3 * k + 1 for k in range(p)])
        K = np.take(ixr, [3 * k + 2 for k in range(p)])

        x, y, z = vertices.T

        mesh3D = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=I,
            j=J,
            k=K,
            flatshading=True,
            colorscale=self.colorscale,
            intensity=z,
            name=target_name,
            showscale=False)

        title = f"STL Model {target_name}"
        layout = get_layout(title)
        fig = go.Figure(data=[mesh3D], layout=layout)

        if save:
            target_path = os.path.join(self.target_dir, target_name + '.html')
            fig.write_html(target_path)
        else:
            fig.show()

    def mesh_checks(self):
        """
        Prints O3D TriangleMesh validity checks

        :param mesh: O3D mesh object
        """
        # tests if all vertices are manifold
        print("all vertices manifold:", self.mesh.is_vertex_manifold())
        # tests if all edges are manifold
        print("all edges manifold:", self.mesh.is_edge_manifold())
        # tests if the mesh is watertight
        print("mesh is watertight:", self.mesh.is_watertight())
