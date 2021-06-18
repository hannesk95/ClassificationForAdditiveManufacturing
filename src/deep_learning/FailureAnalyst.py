import numpy as np
import mlflow
from torch.utils.data import DataLoader


class FailureAnalyst:
    """# TODO: Docstring"""

    def __init__(self, config: object, trainer: object, val_data: object):
        """# TODO: Docstring"""
        self.config = config
        self.trainer = trainer
        self.val_data = val_data
        self.val_dataloader = None

    def start_failure_analysis(self):
        """# TODO: Docstring"""

        # Get true labels
        true_labels = []
        self.val_dataloader = DataLoader(self.val_data, batch_size=len(self.val_data), **self.config.kwargs)
        for idx, label in enumerate(self.val_dataloader):
            true_labels.append(label)

        # Get predicted labels
        pred_labels = self.trainer.validate(val_dataloaders=self.val_dataloader)

        # Compare true labels and predicted labels
        result = np.equal(np.array(true_labels), np.array(pred_labels))

        # Get indices of failed predictions and store respective model path
        failure_idx = np.where(result == False)
        failed_models = []
        for i in failure_idx:
            failed_models.append(self.val_data.dataset.models[i])

        # Store list of failed models using MLflow
        mlflow.log_artifact("list_failed_models", failed_models)
