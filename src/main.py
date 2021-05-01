# A "main" class to call multiple classes together
import numpy as np
import open3d as o3d
from MeshReader import MeshReader
from DataNormalizer import DataNormalizer

def main():
    mesh_reader = MeshReader("stl_mesh.stl")
    mesh = mesh_reader.mesh

    data_normalizer = DataNormalizer("data_normalizer")
    data_normalizer.center_mesh_around_origin(mesh)
    scale_factor = 1 / np.max(mesh.get_max_bound() - mesh.get_min_bound())
    data_normalizer.scale(mesh, scale_factor)
    min_MOI_axis = data_normalizer.min_MOI_axis(mesh)
    coordinate_axis =  np.array([0., 0., 1.])
    rotation_matrix = data_normalizer.align_vectors(mesh, min_MOI_axis, coordinate_axis)
    min_MOI_axis = rotation_matrix.dot(min_MOI_axis)

    # O3D coordinate axes
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # O3D line from origin to axis of min MOI
    points = [[0, 0, 0], min_MOI_axis]
    lines = [ [0, 1] ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.scale(2, [0, 0, 0])

    # visualize mesh after alignment
    mesh_reader.visualize([mesh_frame, line_set])

if __name__ == "__main__":
    main()