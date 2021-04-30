import unittest
from src.visualization import Visualizer
import numpy as np
import os


class MyTestCase(unittest.TestCase):
    def test_plot_voxel(self):
        visualizer = Visualizer('tests')
        test_model = np.array([[[0, 1], [1, 1]], [[0, 1], [1, 1]]])

        visualizer.plot_voxel(test_model, 'test_plot_voxel')

        target_path = 'test_plot_voxel.html'
        self.assertTrue(os.path.exists(target_path))
        os.remove(target_path)


if __name__ == '__main__':
    unittest.main()
