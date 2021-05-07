import numpy as np
import open3d as o3d
from stl import mesh
import pymeshlab
from MeshReader import MeshReader
from MeshSaver import MeshSaver


class DataCleaner():
    def __init__(self,arg):
        self.arg = arg

    # Clean the model using pymeshlab
    def clean(self, vertices, faces):
        """
        Creates
        :param vertices: numpy array containing vertices
        :param faces: numpy array containing faces
        :return: vertices_cleaned: numpy array containing vertices cleaned
                 faces_cleaned: numpy array containing faces cleaned

        """

        ms = pymeshlab.MeshSet()
        # load mesh using vertices and faces
        m = pymeshlab.Mesh(vertices, faces)
        ms.add_mesh(m)
        # apply filter to clean the model
        ms.remove_isolated_folded_faces_by_edge_flip()
        ms.remove_duplicate_faces()
        #get triangles and vertices matrices
        face_matrix = ms.current_mesh().face_matrix()
        vertex_matrix = ms.current_mesh().vertex_matrix()
        # construct numpy arrays
        faces_cleaned =  np.array(face_matrix)
        vertices_cleaned =  np.array(vertex_matrix)
        return vertices_cleaned,faces_cleaned

    def mesh_checks(self, mesh):
        """
        Prints O3D TriangleMesh validity checks

        :param mesh: O3D mesh object
        """
        # tests if all vertices are manifold
        print("all vertices manifold:", mesh.is_vertex_manifold())
        # tests if all edges are manifold
        print("all edges manifold:", mesh.is_edge_manifold())
        # tests if the mesh is watertight
        print("mesh is watertight:", mesh.is_watertight())

    def fix_mesh_vertices(self, mesh):
        """
        Applies O3D TriangleMesh vertices fixes

        :param mesh: O3D mesh object
        """
        # removes vertices that have identical coordinates
        mesh.remove_duplicated_vertices()
        # removes vertices that are not referenced in any triangle
        mesh.remove_unreferenced_vertices()

    def fix_mesh_edges(self, mesh):
        """
        Applies O3D TriangleMesh edges fixes

        :param mesh: O3D mesh object
        """
        # removes all non-manifold edges
        # by successively deleting triangles with the smallest surface area 
        # adjacent to the non-manifold edge until the number of adjacent 
        # triangles to the edge is <= 2
        mesh.remove_non_manifold_edges()

    def fix_mesh_triangles(self, mesh):
        """
        Applies O3D TriangleMesh triangles fixes

        :param mesh: O3D mesh object
        """
        # removes triangles that reference the same three vertices
        mesh.remove_duplicated_triangles()
        # removes triangles that reference a single vertex multiple times in a single triangle
        # usually happens as a result of removing duplicated vertices
        mesh.remove_degenerate_triangles()

##test

if __name__ == "__main__":
    cleaner = DataCleaner("arg")
    mesh_reader = MeshReader("D:/TUM/summer_2021/TUM_DI_LAB/project/project_dev_branch/data/20_20_4_calcube_hole.stl")
    mesh = mesh_reader.mesh
    vertices, faces = cleaner.clean(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    mesh_saver = MeshSaver("D:/TUM/summer_2021/TUM_DI_LAB/project/project_dev_branch/data/cleaned.stl",vertices,faces)
    mesh_saver.save()