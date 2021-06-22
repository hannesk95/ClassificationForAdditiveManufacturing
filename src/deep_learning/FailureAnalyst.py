import numpy as np
import mlflow
import torch


class FailureAnalyst:
    """# TODO: Docstring"""

    def __init__(self, config: object, val_data: object, nn_model: object):
        """# TODO: Docstring"""
        self.config = config
        self.val_data = val_data
        self.nn_model = nn_model

    def start_failure_analysis(self):
        """# TODO: Docstring"""

        # Get true labels & predicted labels
        true_labels = []
        pred_labels = []
        for i in range(len(self.val_data)):
            true_labels.append(self.val_data[i][1])
            pred_labels.append(torch.round(self.nn_model(torch.unsqueeze(self.val_data[i][0], 0))))

        # Compare true labels and predicted labels
        result = np.equal(np.array(true_labels, dtype=int), np.array(pred_labels, dtype=int))

        # Get indices of failed predictions and store respective model path
        failure_idx = list(np.where(result == False)[0])
        failed_models = []
        for i in failure_idx:
            failed_models.append(self.val_data.dataset.models[i])

        # Store list of failed models using MLflow
        mlflow.log_artifact("list_failed_models", failed_models)
