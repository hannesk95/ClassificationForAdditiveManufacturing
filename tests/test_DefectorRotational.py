import unittest
from src.data_generation.transformations import DefectorRotation
import numpy as np
import os
from src.data_generation.VoxelModel import VoxelModel


class MyTestCase(unittest.TestCase):
    def test_adding_defects(self):
        model_path = '/Volumes/My_Passport_SSD/di-lab/SyntheticDataset/256x256x256'
        models = os.listdir(model_path)
        models = [os.path.join(model_path, model) for model in models if model.endswith('.npz')]
        defector = DefectorRotation(hole_radius_nonprintable=5, hole_radius_printable=10, border_nonprintable=3,
                                    border_printable=5, rotation=False, number_of_trials=5,
                                    visualize_top_down_view=False)

        idx = 99
        model_data = np.load(models[idx])['model']
        model = VoxelModel(model_data, np.array([1]), 'test')
        defector(model)


if __name__ == '__main__':
    unittest.main()
