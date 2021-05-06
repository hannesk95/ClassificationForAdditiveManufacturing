import open3d as o3d
from tqdm import tqdm

from src.data_generation.model import Model


class BatchDataProcessor:
    """
    Data generator class which yields data batches to the caller.
    """

    def __init__(self, filepaths: list, batch_size: int, transform: object, target_path: string):
        self.filepaths = filepaths
        self.file = None
        self.batch_size = batch_size
        self.data_batch = []
        self.pointer = 0

    def _load_data_batch(self) -> object:
        for self.file in tqdm(range(self.dataset[self.pointer:(self.pointer + self.batch_size)]),
                              desc=f"[INFO]: Loading data batches of size {self.batch_size}!"):
            self.data_batch.append(Model(self.file))
            self.pointer += self.batch_size

        yield self.data_batch

    def __call__(self) -> object:
        for model in self._load_data_batch():
            if self.transform is not None:
                self.transform(model)
            model.save_as_npz(self.target_path)

