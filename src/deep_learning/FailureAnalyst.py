import numpy as np
import mlflow
import torch
import sys
import os
import logging
from tqdm import tqdm

class FailureAnalyst:
    """# TODO: Docstring"""

    def __init__(self, config: object, val_data: object, nn_model: object):
        """# TODO: Docstring"""
        self.config = config
        self.val_data = val_data
        self.nn_model = nn_model

    def start_failure_analysis(self):
        """# TODO: Docstring"""

        logging.info('Start failure analysis')


        # Get true labels & predicted labels
        true_labels = []
        val_models = []
        for i in tqdm(range(len(self.val_data)), desc="Getting inference data"):
            true_labels.append(self.val_data[i][1])
            val_models.append(self.val_data[i][0])

        models = torch.stack(val_models, dim=0)
        models = models.cuda()
        self.nn_model = self.nn_model.cuda()

        logging.info("Start prediction")
        pred_labels = torch.round(self.nn_model(models))
        logging.info("End prediction")


        # pred_labels.append(torch.round(self.nn_model(torch.unsqueeze(self.val_data[i][0], 0))))

        # Compare true labels and predicted labels
        result = np.equal(np.array(true_labels, dtype=int), pred_labels.detach().numpy().flatten())

        # Get indices of failed predictions and store respective model path
        failure_idx = list(np.where(result == False)[0])
        failed_models = []
        for i in failure_idx:
            failed_models.append(self.val_data.dataset.models[i])

        # Store paths/names of failed models using MLflow
        orig_stdout = sys.stdout
        f = open('failed_models.txt', 'w')
        sys.stdout = f
        for i in range(len(failed_models)):
            print(failed_models[i])
        sys.stdout = orig_stdout
        f.close()
        mlflow.log_artifact("failed_models.txt", artifact_path="failed_models")
        os.remove("failed_models.txt")
