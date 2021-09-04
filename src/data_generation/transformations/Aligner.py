import numpy as np
import open3d as o3d
import pymeshlab


class Aligner:
    """# TODO"""
    def __init__(self, version: int = 1, axis_to_align: np.ndarray = np.array([1., 0., 0.])):
        """# TODO"""
        self.version = version
        self.axis_to_align = axis_to_align

    def __call__(self, model):
        """# TODO"""
        if self.version == 1:
            mesh = model.mesh
            min_MOI_axis = self.min_MOI_axis(mesh)
            self.align_vectors(mesh, min_MOI_axis, self.axis_to_align)
            model.set_model_data(np.asarray(mesh.vertices), np.asarray(mesh.triangle_normals), np.asarray(mesh.triangles))

        elif self.version == 2:
            aligned_vertices, aligned_faces, aligned_normals = self.align(model.mesh.vertices, model.mesh.faces)
            model.set_model_data(aligned_vertices, aligned_faces, aligned_normals)

        return model

    def align(self, mesh):
        """
        align 3D models
        :param mesh model
        :return: mesh vertices,faces,normals aligned
        """

        ms = pymeshlab.MeshSet()
        # load mesh using vertices and faces
        m = pymeshlab.Mesh(mesh.vertices, mesh.faces)
        ms.add_mesh(m)
        # apply filter to align the model to principal component
        ms.transform_align_to_principal_axis()
        # get triangles and vertices matrices
        face_matrix = ms.current_mesh().face_matrix()
        vertex_matrix = ms.current_mesh().vertex_matrix()
        normal_matrix = ms.current_mesh().face_normal_matrix()
        # construct numpy arrays
        faces_aligned = np.array(face_matrix)
        vertices_aligned = np.array(vertex_matrix)
        normals_aligned = np.array(normal_matrix)
        return vertices_aligned, faces_aligned, normals_aligned

    def min_MOI_axis(self, mesh):
        """
        Finds the axis of the minimum moment of interia of a given mesh.
        reference: https://physics.stackexchange.com/questions/426273/how-to-find-the-axis-with-minimum-moment-of-inertia

        :param mesh: O3D mesh object
        :return: axis of the minimum MOI (a numpy (3,) array)
        """
        ##### 1. constructing two arrays: points (mesh vertices) and points squared #####
        points = np.asarray(mesh.vertices)
        points_squared = np.square(points)

        ##### 2. constructing the interia tensor #####
        # horizontal sum of Y and Z components
        I_xx = points_squared[:, 1:].sum(axis=1)
        # vertical sum
        I_xx = I_xx.sum(axis=0)

        # horizontal product of X and Y components
        I_xy = points[:, 0] * points[:, 1]
        # vertical sum
        I_xy = -I_xy.sum(axis=0)

        # horizontal product of X and Z components
        I_xz = points[:, 0] * points[:, 2]
        # vertical sum
        I_xz = -I_xz.sum(axis=0)

        # horizontal sum of X and Z components
        I_yy = points_squared[:, 0] + points_squared[:, 2]
        # vertical sum
        I_yy = I_yy.sum(axis=0)

        # horizontal product of Y and Z components
        I_yz = points[:, 1] * points[:, 2]
        # vertical sum
        I_yz = -I_yz.sum(axis=0)

        # horizontal sum of X and Y components
        I_zz = points_squared[:, 0] + points_squared[:, 1]
        # vertical sum
        I_zz = I_zz.sum(axis=0)

        ##### 3. Eigenvalue decomposition of the inertia tensor#####
        inertia_tensor = np.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])
        eigen_values, eigen_vectors = np.linalg.eigh(inertia_tensor)
        min_eigen_value_index = eigen_values.argmin()
        min_MOI_axis = eigen_vectors[:, min_eigen_value_index]

        return min_MOI_axis

    def align_vectors(self, mesh, mesh_axis, coordinate_axis):
        """
        Aligns a vector within the mesh with one of the coordinate axes.
        reference: https://physics.stackexchange.com/questions/426273/how-to-find-the-axis-with-minimum-moment-of-inertia

        Changes the alignment of the mesh in-place

        :param mesh: O3D mesh object
        :param mesh_axis: mesh axis to be aligned with a coordinate axis (a numpy (3,) arrray)
        :param coordinate_axis: a coordinate axis {[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]} (a numpy (3,) arrray)
        :return: rotation matirx to be applied on the mesh points to apply the alignment (a numpy (3, 3) array)
        """
        ##### 1. constructing a rotation matrix #####
        a = mesh_axis
        b = coordinate_axis
        a = a / np.linalg.norm(a) # normalize a
        b = b / np.linalg.norm(b) # normalize b
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b)

        v1, v2, v3 = v
        h = 1 / (1 + c)

        Vmat = np.array([[0, -v3, v2],
                    [v3, 0, -v1],
                    [-v2, v1, 0]])

        R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)

        ##### 2. applying the rotation via matrix multiplication #####
        vertices_rotated = np.asarray(mesh.vertices).dot(R)
        mesh.vertices = o3d.utility.Vector3dVector(vertices_rotated)

        return R
