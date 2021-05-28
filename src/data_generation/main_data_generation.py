import sys
sys.path.append(".")   #TODO Ugly - currently needed for LRZ AI System - find better solution
sys.path.append("..")
sys.path.append("../..")
import logging
from src.data_generation.ParamConfigurator import ParamConfigurator
from src.data_generation.ModelSelector import ModelSelector
from src.data_generation.BatchDataProcessor import BatchDataProcessor
from src.data_generation.transformations import Normalizer, Aligner, Cleaner, Voxelizer, VoxelizerGPU, DefectorRotation, \
    ComposeTransformer


def main():
    # 1. Define configuration parameters
    config = ParamConfigurator()

    # 2. Preselect files
    selector = ModelSelector(input_path=config.model_path, max_filesize=config.max_filesize,
                             min_compactness=config.min_compactness, num_files=config.num_files)
    final_models = selector.select_models()

    # 3. Define transformations
    normalizer = Normalizer()
    aligner = Aligner()
    cleaner = Cleaner()
    voxelizer = Voxelizer(dimension=config.voxel_dimensions, representation=config.voxel_representation)
    defector = DefectorRotation(radius=config.hole_radius, border=config.border)

    # 4. Compose transformations
    composer = ComposeTransformer([cleaner, normalizer, aligner, voxelizer, defector])

    # 5. Start processing using batch of files
    batch_processor = BatchDataProcessor(final_models, batch_size=config.batch_size, transformer=composer,
                                         target_path=config.target_path)
    batch_processor.process()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_data_generation')

    main()

    logging.info('Finished main_data_generation')
