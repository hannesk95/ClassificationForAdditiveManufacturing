from tqdm import tqdm
import numpy as np
from src.data_generation.MeshModel import MeshModel


class BatchDataProcessor:
    """Data generator class which yields data batches to the caller."""

    def __init__(self, filepaths: list, batch_size: int, transformer: object, target_path: str):
        """# TODO"""
        self.filepaths = filepaths
        self.file = None
        self.batch_size = batch_size
        self.pointer = 0
        self.transformer = transformer
        self.target_path = target_path

    def _load_data_batch(self):
        """# TODO"""
        for _ in range(int(np.floor(len(self.filepaths) / self.batch_size))):
            data_batch = []
            for self.file in self.filepaths[self.pointer:(self.pointer + self.batch_size)]:
                data_batch.append(MeshModel(self.file))
            self.pointer += self.batch_size

            yield data_batch

        data_batch = []
        if self.pointer != len(self.filepaths):
            for self.file in self.filepaths[self.pointer:len(self.filepaths)]:
                data_batch.append(MeshModel(self.file))

            yield data_batch

    def _save_model(self, models):
        """# TODO"""
        if type(models) is list:
            for model_instance in models:
                model_instance.save(self.target_path)
        else:
            models.save(self.target_path)

    def process(self):
        """# TODO"""
        for batch in tqdm(self._load_data_batch(), total=int(np.floor(len(self.filepaths) / self.batch_size))+1,
                          desc="INFO - Batch: "):
            for model in tqdm(batch, desc="INFO - Running models through the pipeline"):
                if self.transformer is not None:
                    models = self.transformer(model)
                if models is not None:
                    self._save_model(models)
