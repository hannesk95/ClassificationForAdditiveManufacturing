import os
import subprocess
import open3d as o3d
from pathlib import Path
from src.data_generation.utils import binvox2npz


def _convert_stl2obj(model: object) -> str:
    """ Internal method in order to convert .stl files into .obj files.
    This is needed as the CUDA voxelizer requires .obj input file format. """

    mesh_stl = model.mesh
    mesh_obj_path = str(Path(model.path).with_suffix('.obj'))
    o3d.io.write_triangle_mesh(mesh_obj_path, mesh_stl)

    return mesh_obj_path


class VoxelizerGPU:

    def __init__(self, dimension: int = 128):
        """ # TODO """
        self.dimension = dimension

    def __call__(self, model: object):
        """ Call method which starts the CUDA voxelizer. Depending on the input file format,
        the input file is either converted first into .obj file format, followed by a blocking SHELL
        command execution or done straight forward. Finally, the resulting .binvox file is converted
        into a numpy array and stored in a compressed file format."""

        model_path = model.path

        if model.path.endswith(".stl"):
            model_path = _convert_stl2obj(model)
            cmd, voxel_model_path = self._get_shell_command(model_path)
            subprocess.call(cmd, shell=True)
            model = binvox2npz(str(Path(voxel_model_path).with_suffix('.binvox')))
            os.remove(model_path)

        else:
            cmd = self._get_shell_command(model_path)
            subprocess.call(cmd, shell=True)
            model = binvox2npz(Path(model_path).with_suffix('.binvox'))

        return model

    def _get_shell_command(self, model_path: str) -> list:
        """ Internal method in order to prepare the shell command to be executed in order
        to start the CUDA voxelizer. """

        path_voxelizer = "/voxelizer/build/bin/voxelizer "  # Do not change this path!
        resolution = f"-r {self.dimension} "
        path_input = model_path + " "
        path_output = str(Path(model_path).with_suffix('')) + str(self.dimension)

        return [path_voxelizer + resolution + path_input + path_output, path_output]
