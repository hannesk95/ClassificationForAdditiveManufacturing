
import numpy as np
import open3d as o3d
from stl import mesh
import pymeshlab


#####align the model using pymeshlab

class DataNormalizer():
    def __init__(self,arg):
        self.arg = arg

    def align(self, stl_path, target_path=None):
        """
        Creates a 3D interactive JavaScript plot of a given stl 3D model
        :param stl_path: Source path to the stl model
        :param target_path: path of the resulting stl file.
        :return: stl file containing the model aligned
        """

        ms = pymeshlab.MeshSet()
        # load mesh
        ms.load_new_mesh(stl_path)
        # apply filter to align the model to principal component
        ms.transform_align_to_principal_axis()
        ms.save_current_mesh(target_path)


normalize = DataNormalizer("arg")
##test
normalize.align("20_20_4_calcube_hole.stl","aligned.stl")