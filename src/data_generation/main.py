import sys
sys.path.append(".")   #TODO Ugly - currently needed for LRZ AI System - find better solution
sys.path.append("..")
sys.path.append("../..")
import logging
import torch
from src.data_generation.ParamConfigurator import ParamConfigurator
from src.data_generation.ModelSelector import ModelSelector
from src.data_generation.BatchDataProcessor import BatchDataProcessor
from src.data_generation.transformations import Normalizer, Aligner, Cleaner, Voxelizer, VoxelizerGPU, Defector, \
    DefectorTopDownView, DefectorRotation, ComposeTransformer
from src.data_generation import utils


def main():
    # 1. Define configuration parameters
    config = ParamConfigurator()

    # 2. Preselect files
    selector = ModelSelector(input_path=config.model_path, max_filesize=config.max_filesize,
                             min_compactness=config.min_compactness, num_files=config.num_files)
    final_models = selector.select_models()

    # 3. Define transformations
    normalizer = Normalizer()
    aligner = Aligner(version=config.version, axis_to_align=config.axis_to_align)
    cleaner = Cleaner()
    if torch.cuda.is_available():
        voxelizer = VoxelizerGPU(dimension=config.voxel_dimensions)
    else:
        voxelizer = Voxelizer(dimension=config.voxel_dimensions, representation=config.voxel_representation)

    if config.defector_type == 'DefectorTopDownView':
        defector = DefectorTopDownView(hole_radius_nonprintable=config.hole_radius_nonprintable,
                                       hole_radius_printable=config.hole_radius_printable,
                                       border_nonprintable=config.border_nonprintable,
                                       border_printable=config.border_printable,
                                       number_of_trials=config.number_of_trials)
    elif config.defector_type == 'DefectorExhaustive':
        defector = Defector(max_cylinder_diameter = config.max_cylinder_diameter,
                            trials = config.trials,
                            remaining_voxels = config.remaining_voxels,
                            factor = config.factor)
    elif config.defector_type == 'DefectorRotation':
        defector = DefectorRotation(hole_radius_nonprintable=config.hole_radius_nonprintable_rot,
                                       hole_radius_printable=config.hole_radius_printable_rot,
                                       border_nonprintable=config.border_nonprintable_rot,
                                       border_printable=config.border_printable_rot,
                                       number_of_trials=config.number_of_trials_rot)
        pass 

    # 4. Compose transformations
    composer = ComposeTransformer([cleaner, normalizer, aligner, voxelizer, defector])

    # 5. Start processing using batch of files
    batch_processor = BatchDataProcessor(final_models, batch_size=config.batch_size, transformer=composer,
                                         target_path=config.target_path)
    batch_processor.process()

    utils.plot_all_models(source_path=config.target_path, cutoff=50, skip_images=True)


if __name__ == '__main__':
    logging.basicConfig(filename='main_data_generation.log', encoding='utf-8',
                        format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_data_generation')

    main()

    logging.info('Finished main_data_generation')
