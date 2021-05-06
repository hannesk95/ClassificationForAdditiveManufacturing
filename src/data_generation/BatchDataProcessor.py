import open3d as o3d
from tqdm import tqdm
from src.data_generation.Model import Model


class BatchDataProcessor:
    """
    Data generator class which yields data batches to the caller.
    """

    def __init__(self, filepaths: list, batch_size: int, transformer: object, target_path: string):
        self.filepaths = filepaths
        self.file = None
        self.batch_size = batch_size
        self.data_batch = []
        self.pointer = 0
        self.transformer = transformer

    def _load_data_batch(self) :
        for self.file in tqdm(range(self.filepaths[self.pointer:(self.pointer + self.batch_size)]),
                              desc=f"[INFO]: Loading data batches of size {self.batch_size}!"):
            self.data_batch.append(Model(self.file))
            self.pointer += self.batch_size

        yield self.data_batch

    def process(self):
        for model in self._load_data_batch():
            if self.transformer is not None:
                self.transformer(model)
            model.save_as_npz(self.target_path)







