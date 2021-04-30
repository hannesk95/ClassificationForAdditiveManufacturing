class DataPreprocessor:
    def __init__(self, dataset, batch_size, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.process()

    def process(self):
        pass
