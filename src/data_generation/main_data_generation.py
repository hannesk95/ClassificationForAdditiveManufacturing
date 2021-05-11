import os
import logging
import configparser
from src.data_generation.ModelSelector import ModelSelector
from src.data_generation.BatchDataProcessor import BatchDataProcessor
from src.data_generation.transformations.Normalizer import Normalizer
from src.data_generation.transformations.Aligner import Aligner
from src.data_generation.transformations.Voxelizer import Voxelizer
from src.data_generation.transformations.Defector import Defector
from src.data_generation.transformations.ComposeTransformer import ComposeTransformer


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    model_path = config['filepaths']['model_path']
    if model_path is None:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Sample_Data'))

    target_path = config['filepaths']['target_path']
    if target_path is None:
        target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'SyntheticDataset'))
        if not os.path.exists(target_path):
            os.makedirs(target_path)
    
    num_files = config['ModelSelector'].getint('num_files')
    max_filesize = config['ModelSelector'].getfloat('max_filesize')
    min_compactness = config['ModelSelector'].getfloat('min_compactness')
    batch_size = config['BatchDataProcessor'].getint('batch_size')

    # 1. Preselect files
    selector = ModelSelector(input_path=model_path, max_filesize=max_filesize,
                             min_compactness=min_compactness, num_files=num_files)
    final_models = selector.select_models()

    # 2. Define transformations
    normalizer = Normalizer()
    aligner = Aligner()
    voxelizer = Voxelizer()
    defector = Defector()

    # 3. Compose transformations
    composer = ComposeTransformer([normalizer, aligner, voxelizer, defector])

    # 4. Start processing using batch of files
    batch_processor = BatchDataProcessor(final_models, batch_size=batch_size, transformer=composer)
    batch_processor.process()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_data_generation')

    main()

    logging.info('Finished main_data_generation')
