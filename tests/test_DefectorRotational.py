import unittest
from src.data_generation.transformations import DefectorRotation
import numpy as np
import os
from src.data_generation.VoxelModel import VoxelModel


class MyTestCase(unittest.TestCase):
    def test_adding_defects(self):
        model_path = '/Volumes/My_Passport_SSD/di-lab/SyntheticDataset/voxel_data-3'
        models = os.listdir(model_path)
        models = [os.path.join(model_path, model) for model in models if model.endswith('.npz')]
        defector = DefectorRotation(10, 15, False, 5, True)

        idx = 0
        model = VoxelModel(np.load(models[idx])['model'], np.array([1]), models[idx])
        model_data = model.model

        defector(model)

if __name__ == '__main__':
    unittest.main()
