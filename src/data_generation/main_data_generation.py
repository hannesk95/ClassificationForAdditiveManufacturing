import logging
from src.data_generation.Configurator import Configurator
from src.data_generation.ModelSelector import ModelSelector
from src.data_generation.BatchDataProcessor import BatchDataProcessor
from src.data_generation.transformations import Normalizer, Aligner, Voxelizer, VoxelizerGPU, Defector, ComposeTransformer


def main():

    # 1. Define configuration parameters
    config = Configurator()

    # 2. Preselect files
    selector = ModelSelector(input_path=config.model_path, max_filesize=config.max_filesize,
                             min_compactness=config.min_compactness, num_files=config.num_files)
    final_models = selector.select_models()

    # 3. Define transformations
    # normalizer = Normalizer()
    # aligner = Aligner()
    voxelizer = VoxelizerGPU()
    # defector = Defector()

    # 4. Compose transformations
    # composer = ComposeTransformer([normalizer, aligner, voxelizer, defector])

    # 5. Start processing using batch of files
    batch_processor = BatchDataProcessor(final_models, batch_size=config.batch_size, transformer=voxelizer,
                                         target_path=config.target_path)
    batch_processor.process()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_data_generation')

    main()

    logging.info('Finished main_data_generation')
