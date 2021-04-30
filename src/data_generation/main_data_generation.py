import logging
import os
from transformations import Voxelizer, Normalizer, ComposeTransformer
from dataset import Dataset
from data_preprocessor import DataPreprocessor


def main():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    dataset = Dataset(model_path)
    dataset[0]
    pass


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_data_generation')

    main()

    logging.info('Finished main_data_generation')
