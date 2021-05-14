import os
import subprocess
import open3d as o3d
from pathlib import Path
from src.data_generation.utils import binvox2npz


class VoxelizerGPU:

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def __call__(self, model: object):
        """ Call method which starts the CUDA voxelizer. Depending on the input file format,
        the input file is either converted first into .obj file format, followed by a blocking SHELL
        command execution or done straight forward. Finally, the resulting .binvox file is converted
        into a numpy array and stored in a compressed file format."""

        model_path = model.path

        if model.path.endswith(".stl"):
            model_path = self._convert_stl2obj(model_path)
            cmd = self._get_shell_command(model_path)
            subprocess.call(cmd, shell=True)
            binvox2npz(Path(model_path).with_suffix('.binvox'))
            os.remove(model_path)

        else:
            cmd = self._get_shell_command(model_path)
            subprocess.call(cmd, shell=True)
            binvox2npz(Path(model_path).with_suffix('.binvox'))

    def _convert_stl2obj(self, model_path: str) -> str:
        """ Internal method in order to convert .stl files into .obj files.
        This is needed as the CUDA voxelizer requires .obj input file format. """

        mesh_stl = o3d.io.read_triangle_mesh(model_path)
        mesh_obj_path = Path(model_path).with_suffix('.obj')
        o3d.io.write_triangle_mesh(mesh_obj_path, mesh_stl)

        return mesh_obj_path

    def _get_shell_command(self, model_path: str) -> str:
        """ Internal method in order to prepare the shell command to be executed in order
        to start the CUDA voxelizer. """

        path_voxelizer = "./build/bin/voxelizer "
        resolution = f"-r {self.dimension} "
        path_input = model_path
        path_output = os.path.join(Path(model_path).with_suffix(''), "_voxelized_", str(self.dimension))

        return os.path.join(path_voxelizer, resolution, path_input, path_output)
