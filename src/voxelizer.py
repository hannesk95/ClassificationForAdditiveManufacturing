from mesh_to_sdf import mesh_to_voxels

import trimesh
from skimage import measure

class DataConverter():
    def __init__(self,input_mesh_file_path):
        self.input_mesh_file_path = input_mesh_file_path

    def voxelize(self):
        """
        The function converts an input triangular mesh into a voxelized array.
        :input_mesh_file_path : Provide the input path to the .stl file or .obj file
        :return : N x N x N SDF values
        """
        self.mesh = trimesh.load(self.input_mesh_file_path)

        self.voxels = mesh_to_voxels(self.mesh, 64, sign_method='depth', pad=False)

        vertices, faces, normals, _ = measure.marching_cubes(self.voxels, level=0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        mesh.show()


#Testing 
if __name__ == '__main__':
    voxel_rep = DataConverter('/home/aditya/Desktop/FPES_Engine_Nozzles.STL')
    voxel_rep.voxelize()

