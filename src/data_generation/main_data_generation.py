import logging
import os
from transformations import Normalizer, Aligner, Voxelizer, Defector, ComposeTransformer
from dataset import Dataset
from data_preprocessor import DataPreprocessor

from data_generation import BatchDataProcessor, ModelSelector


def main():
    # Location of our dataset
    # TODO create config and read config here
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    target_path =
    max_file_number =
    max_filesize =
    min_compactness =
    batch_size =

    # 1. Preselect files
    model_selector = ModelSelector(input_path=model_path, max_filesize=1, min_compactness=None)
    final_models = model_selector()

    # 2. Define transformations
    normalizer = Normalizer()
    aligner = Aligner()
    voxelizer = Voxelizer()
    defector = Defector()

    # 3. Compose transformations
    composer = ComposeTransformer([normalizer, aligner, voxelizer, defector])

    # 4. Start loading data in batches
    loader = BatchDataProcessor(final_models, batch_size=500, transform=composer)
    loader()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_data_generation')

    main()

    logging.info('Finished main_data_generation')
