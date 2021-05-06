import logging
import os
from transformations import Voxelizer, Normalizer, ComposeTransformer
from dataset import Dataset
from data_preprocessor import DataPreprocessor

from data_generation import BatchDataLoader, ModelSelector


def main():

    # Location of our dataset
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

    # 1. Preselect files
    model_selector = ModelSelector(input_path=model_path, max_filesize=1, min_compactness=None)
    final_models = model_selector()

    # 2. Start loading data in batches
    loader = BatchDataLoader(filepaths=final_models, batch_size=500)

    for model in loader():
        pass

        # 3. Align model

        # 4. Normalize model

        # 5. Voxelize model + save model + save label "printable)

        # 6. Add defect to model + save model + save label "not printable"

    dataset = Dataset(model_path)
    dataset[0]
    pass


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_data_generation')

    main()

    logging.info('Finished main_data_generation')
