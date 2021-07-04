import numpy as np
import mlflow
import torch
import sys
import os
import logging
from tqdm import tqdm
import horovod.torch as hvd
from sklearn.metrics import confusion_matrix, roc_curve, auc


class FailureAnalyst:
    """# TODO: Docstring"""

    def __init__(self, config: object, val_data: object, nn_model: object, trainer: object, val_dataloader: object):
        """# TODO: Docstring"""
        self.config = config
        self.val_data = val_data
        self.nn_model = nn_model.nn_model
        self.trainer = trainer
        self.val_dataloader = val_dataloader

    def start_failure_analysis(self):
        """# TODO: Docstring"""

        with torch.no_grad():

            logging.info('Start failure analysis')

            self.nn_model.eval()

            # predictions = self.trainer.predict(self.nn_model, self.val_dataloader)
            #
            # print("#######################################")
            # print("Start new prediction")
            # print(predictions)
            # print("End new prediction")
            # print("#######################################")

            # Get true labels & predicted labels
            true_labels = []
            val_models = []
            pred_labels = []
            prob_labels = []
            for i in tqdm(range(len(self.val_data)), desc="Performing failure analysis"):
                true_labels.append(self.val_data[i][1])
                # val_models.append(self.val_data[i][0])

                score = self.nn_model(torch.unsqueeze(self.val_data[i][0], 0))
                prob_labels.append(score)
                pred_labels.append(torch.round(score))
                # pred_labels.append(torch.round(self.nn_model(torch.unsqueeze(self.val_data[i][0], 0))))

            # Compute confusion matrix and store results using MLflow
            tn, fp, fn, tp = confusion_matrix(np.array(true_labels, dtype=int), torch.Tensor(pred_labels).numpy()).ravel()
            mlflow.log_param("true_negative", tn)
            mlflow.log_param("false_positive", fp)
            mlflow.log_param("false_negative", fn)
            mlflow.log_param("true_positive", tp)

            # Compute ROC/AUC and store results using MLflow
            fpr, tpr, _ = roc_curve(np.array(true_labels, dtype=int), torch.Tensor(prob_labels).numpy())
            roc_auc = auc(fpr, tpr)
            # mlflow.log_artifact("false_positive_rate", fpr)
            # mlflow.log_artifact("true_positive_rate", tpr)
            mlflow.log_param("area_under_curve", roc_auc)

            # models = torch.stack(val_models, dim=0)

            # Compare true labels and predicted labels
            # result = np.equal(np.array(true_labels, dtype=int), pred_labels.detach().numpy().flatten())
            result = np.equal(np.array(true_labels, dtype=int), torch.Tensor(pred_labels).numpy())

            # Get indices of failed predictions and store respective model path
            failure_idx = list(np.where(result == False)[0])
            failed_models = []
            for i in failure_idx:
                failed_models.append(self.val_data.dataset.models[i])

            if hvd.rank() == 0:
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
