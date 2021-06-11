import os
import configparser
import numpy as np


class ParamConfigurator:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        self.model_path = config['filepaths']['model_path']
        if self.model_path == 'None':
            self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Sample_Data'))

        self.target_path = config['filepaths']['target_path']
        if self.target_path == 'None':
            self.target_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'SyntheticDataset'))
        if not os.path.exists(self.target_path):
            os.makedirs(self.target_path)

        self.num_files = config['ModelSelector'].getint('num_files')
        self.max_filesize = config['ModelSelector'].getfloat('max_filesize')
        self.min_compactness = config['ModelSelector'].getfloat('min_compactness')
        self.batch_size = config['BatchDataProcessor'].getint('batch_size')

        self.voxel_dimensions = config['Voxelization'].getint('voxel_dimensions')
        self.voxel_representation = config['Voxelization']['voxel_representation']

        self.hole_radius_nonprintable = config['Defector'].getint('hole_radius_nonprintable')
        self.hole_radius_printable = config['Defector'].getint('hole_radius_printable')
        self.border_nonprintable = config['Defector'].getint('border_nonprintable')
        self.border_printable = config['Defector'].getint('border_printable')
        self.rotation = config['Defector'].getboolean('rotation')
        self.number_of_trials = config['Defector'].getint('number_of_trials')

        self.max_cylinder_diameter = config['ModelDefector'].getint('max_cylinder_diameter')
        self.trials = config['ModelDefector'].getint('trials')
        self.remaining_voxels = config['ModelDefector'].getint('remaining_voxels')
        self.factor = config['ModelDefector'].getfloat('factor')

        self.version = config['Aligner'].getint('version')
        axis_to_align = config['Aligner']['axis_to_align']
        if axis_to_align == 'x':
            self.axis_to_align = np.array([1., 0., 0.])
        elif axis_to_align == 'y':
            self.axis_to_align = np.array([0., 1., 0.])
        elif axis_to_align == 'z':
            self.axis_to_align = np.array([0., 0., 1.])
