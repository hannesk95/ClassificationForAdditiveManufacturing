import open3d as o3d
from tqdm import tqdm


class BatchDataLoader:
    """
    Data generator class which yields data batches to the caller.
    """

    def __init__(self, filepaths: list, batch_size: int):
        self.filepaths = filepaths
        self.file = None
        self.batch_size = batch_size
        self.data_batch = []
        self.pointer = 0

    def _load_data_batch(self) -> object:
        for self.file in tqdm(range(self.filepaths[self.pointer:(self.pointer + self.batch_size)]),
                              desc=f"[INFO]: Loading data batches of size {self.batch_size}!"):
            self.data_batch.append(o3d.io.read_triangle_mesh(self.file))
            self.pointer += self.batch_size

        yield self.data_batch

    def __call__(self) -> object:
        yield self._load_data_batch()
