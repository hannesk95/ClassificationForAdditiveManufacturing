import numpy as np
import pymeshlab

# reference: https://support.shapeways.com/hc/en-us/articles/360007107674-Tips-for-successful-modeling


class Cleaner:
    """# TODO"""
    def __call__(self, model):
        """# TODO"""
        # PyMeshLab
        cleaned_vertices, cleaned_faces, cleaned_normals = self.clean(model)
        model.set_model_data(cleaned_vertices, cleaned_normals, cleaned_faces)
        # O3D
        mesh = model.mesh
        self.fix_mesh_vertices(mesh)
        self.fix_mesh_edges(mesh)
        self.fix_mesh_triangles(mesh)
        model.set_model_data(np.asarray(mesh.vertices), np.asarray(mesh.triangle_normals), np.asarray(mesh.triangles))

        return model

    def clean(self, mesh):
        """
        Clean the model using pymeshlab
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
        # get triangles,vertices,normals matrices
        face_matrix = ms.current_mesh().face_matrix()
        vertex_matrix = ms.current_mesh().vertex_matrix()
        normal_matrix = ms.current_mesh().face_normal_matrix()
        # construct numpy arrays
        faces_cleaned = np.array(face_matrix)
        vertices_cleaned = np.array(vertex_matrix)
        normals_cleaned = np.array(normal_matrix)
        return vertices_cleaned, faces_cleaned, normals_cleaned

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
