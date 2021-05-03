import numpy as np
import open3d as o3d
from stl import mesh
import pymeshlab
from MeshReader import MeshReader
from MeshSaver import MeshSaver


#####Clean the model using pymeshlab

class DataCleaner():
    def __init__(self,arg):
        self.arg = arg

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

##test

if __name__ == "__main__":
    cleaner = DataCleaner("arg")
    mesh_reader = MeshReader("D:/TUM/summer_2021/TUM_DI_LAB/project/project_dev_branch/data/20_20_4_calcube_hole.stl")
    mesh = mesh_reader.mesh
    vertices, faces = cleaner.clean(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    mesh_saver = MeshSaver("D:/TUM/summer_2021/TUM_DI_LAB/project/project_dev_branch/data/cleaned.stl",vertices,faces)
    mesh_saver.save()