import os
import configparser


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

        self.hole_radius = config['Defector'].getint('hole_radius')
        self.border = config['Defector'].getint('border')


