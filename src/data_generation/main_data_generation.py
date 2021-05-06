import os
import logging
from src.data_generation.ModelSelector import ModelSelector
from src.data_generation.BatchDataProcessor import BatchDataProcessor
from src.data_generation.transformations.Normalizer import Normalizer
from src.data_generation.transformations.Aligner import Aligner
from src.data_generation.transformations.Voxelizer import Voxelizer
from src.data_generation.transformations.Defector import Defector
from src.data_generation.transformations.ComposeTransformer import ComposeTransformer


def main():
    # Location of our dataset
    # TODO create config and read config here
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    # target_path =
    # max_file_number =
    # max_filesize =
    # min_compactness =
    # batch_size =

    # 1. Preselect files
    selector = ModelSelector(input_path=model_path, max_filesize=1,
                             min_compactness=0.0, num_files=0)
    final_models = selector.select_models()

    # 2. Define transformations
    normalizer = Normalizer()
    aligner = Aligner()
    voxelizer = Voxelizer()
    defector = Defector()

    # 3. Compose transformations
    composer = ComposeTransformer([normalizer, aligner, voxelizer, defector])

    # 4. Start processing using batch of files
    batch_processor = BatchDataProcessor(final_models, batch_size=500, transformer=composer)
    batch_processor.process()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_data_generation')

    main()

    logging.info('Finished main_data_generation')
