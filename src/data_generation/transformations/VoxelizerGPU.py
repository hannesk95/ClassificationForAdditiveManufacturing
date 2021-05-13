import os
import subprocess
import open3d as o3d
from pathlib import Path
from src.data_generation.utils import binvox2npz
from src.data_generation.utils import extract_file_name


class VoxelizerGPU:

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def __call__(self, model: object):

        model_path = model.path

        if model.path.endswith(".stl"):
            model_path = self._convert_stl2obj(model)

        cmd = self._get_shell_command(model_path)
        subprocess.call(cmd)
        binvox2npz(Path(model_path).with_suffix('.binvox'))
        os.remove(model_path)

    def _convert_stl2obj(self, model: object) -> str:
        mesh_stl = o3d.io.read_triangle_mesh(model.path)
        mesh_obj_path = Path(model.path).with_suffix('.obj')
        o3d.io.write_triangle_mesh(mesh_obj_path, mesh_stl)

        return mesh_obj_path

    def _get_shell_command(self, model_path: str) -> str:

        path_voxelizer = "./build/bin/voxelizer "
        resolution = f"-r {self.dimension} "
        path_input = model_path
        path_output = os.path.join(Path(model_path).with_suffix(''), "_voxelized_", str(self.dimension))

        return os.path.join(path_voxelizer, resolution, path_input, path_output)

        # "./voxelizer -r 64 ./data/sphere/sphere.obj ./data/sphere/sphere_voxelized_64"
