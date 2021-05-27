import numpy as np
import open3d as o3d
from stl import mesh
import pymeshlab


# reference: https://support.shapeways.com/hc/en-us/articles/360007107674-Tips-for-successful-modeling

class DataCleaner():
    def __init__(self,mesh):
        self.mesh = mesh

    def __call__(self):
        cleaned_vertices, cleaned_faces, cleaned_normals = self.clean(self.mesh.vertices,self.mesh.faces)
        self.mesh.set_model_data(cleaned_vertices, cleaned_faces, cleaned_normals)
        self.mesh_checks(self.mesh)
        self.fix_mesh_vertices(self.mesh)
        self.fix_mesh_edges(self.mesh)
        self.fix_mesh_triangles(self.mesh)
        self.mesh.set_model_data(np.asarray(self.mesh.vertices), np.asarray(self.mesh.triangle_normals),np.asarray(self.mesh.triangles))

    # Clean the model using pymeshlab
    def clean(self, mesh):
        """
        Creates
        :param mesh model
        :return: vertices_cleaned: numpy array containing vertices cleaned
                 faces_cleaned: numpy array containing faces cleaned
                 normals_cleaned: numpy array containing faces cleaned

        """

        ms = pymeshlab.MeshSet()
        # load mesh using vertices and faces
        m = pymeshlab.Mesh(mesh.vertices, mesh.faces)
        ms.add_mesh(m)
        # apply filter to clean the model
        ms.remove_isolated_folded_faces_by_edge_flip()
        ms.remove_duplicate_faces()
        #get triangles,vertices,normals matrices
        face_matrix = ms.current_mesh().face_matrix()
        vertex_matrix = ms.current_mesh().vertex_matrix()
        normal_matrix = ms.current_mesh().face_normal_matrix()
        # construct numpy arrays
        faces_cleaned =  np.array(face_matrix)
        vertices_cleaned =  np.array(vertex_matrix)
        normals_cleaned = np.array(normal_matrix)
        return vertices_cleaned,faces_cleaned,normals_cleaned

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
