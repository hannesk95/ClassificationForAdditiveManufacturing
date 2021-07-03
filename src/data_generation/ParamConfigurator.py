import os
import configparser
import numpy as np
from shutil import copyfile


class ParamConfigurator:

    def __init__(self, config_path: str='../../infra/data_generation/config.ini'):
        config = configparser.ConfigParser()
        config.read(config_path)

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

        self.defector_type = config['Defector']['defector_type']

        self.hole_radius_nonprintable = config['DefectorTopDownView'].getint('hole_radius_nonprintable')
        self.hole_radius_printable = config['DefectorTopDownView'].getint('hole_radius_printable')
        self.border_nonprintable = config['DefectorTopDownView'].getint('border_nonprintable')
        self.border_printable = config['DefectorTopDownView'].getint('border_printable')
        self.rotation = config['DefectorTopDownView'].getboolean('rotation')
        self.number_of_trials = config['DefectorTopDownView'].getint('number_of_trials')

        self.max_cylinder_diameter = config['ModelDefector'].getint('max_cylinder_diameter')
        self.trials = config['ModelDefector'].getint('trials')
        self.remaining_voxels = config['ModelDefector'].getint('remaining_voxels')
        self.factor = config['ModelDefector'].getfloat('factor')
        
        self.hole_radius_nonprintable_rot = config['DefectorRotation'].getint('hole_radius_nonprintable_rot')
        self.hole_radius_printable_rot = config['DefectorRotation'].getint('hole_radius_printable_rot')
        self.border_nonprintable_rot = config['DefectorRotation'].getint('border_nonprintable_rot')
        self.border_printable_rot = config['DefectorRotation'].getint('border_printable_rot')
        self.number_of_trials_rot = config['DefectorRotation'].getint('number_of_trials_rot')
        
        
        
        
        
        self.version = config['Aligner'].getint('version')
        axis_to_align = config['Aligner']['axis_to_align']
        if axis_to_align == 'x':
            self.axis_to_align = np.array([1., 0., 0.])
        elif axis_to_align == 'y':
            self.axis_to_align = np.array([0., 1., 0.])
        elif axis_to_align == 'z':
            self.axis_to_align = np.array([0., 0., 1.])

        copyfile(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)),
                 os.path.join(self.target_path, 'used_config.ini'))
