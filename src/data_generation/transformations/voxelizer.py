from mesh_to_sdf import mesh_to_voxels
import numpy as np
import trimesh
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DataConverter():
    def __init__(self,input_mesh_file_path):
        self.input_mesh_file_path = input_mesh_file_path
        

    def meshtosdf(self):
        """
        The function converts an input triangular mesh into a voxelized array.
        """
        
        mesh = trimesh.load(self.input_mesh_file_path)
        voxels = mesh_to_voxels(mesh, 64, sign_method='depth', pad=False)
        return voxels
        
    def visualise_sdf(self,sdf):
        """
        The function reconstructs the mesh using marching cubes algorithm and renders it.
        : sdf: N x N x N array of sdf values where N is the number of voxels.
        """
        vertices, faces, normals, _ = measure.marching_cubes(sdf, level=0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        mesh.show()

        
    def sdftobinary(self,sdf):
        """
        The function takes in sdf values and converts it into a binary field.
        : sdf: N x N x N array of sdf values where N is the number of voxels.
        
        """

        binay_voxels = np.ones_like(sdf)
        dims = [sdf.shape[0],sdf.shape[1],sdf.shape[2]]

        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    if (abs(float(sdf[i][j][k])) > 0.05):
                        binay_voxels[i][j][k] = 0
        np.savez_compressed("occupancy_grid.npz",voxels=binay_voxels) #To load use np.load("<Path to the file occupancy_grid.npz>")['voxels']
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(binay_voxels)
        plt.show()        
        


#Testing 
if __name__ == '__main__':
    voxel_rep = DataConverter('/home/aditya/Documents/4th_sem/DI_Lab/Code/Stanford_Bunny.stl')
    sdf_val = voxel_rep.meshtosdf()
    voxel_rep.visualise_sdf(sdf_val)
    #voxel_rep.sdftobinary(sdf_val)

