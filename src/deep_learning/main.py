import sys

import torch

sys.path.append(".")   #TODO Ugly - currently needed for LRZ AI System - find better solution
sys.path.append("..")
sys.path.append("../..")

import logging
import mlflow.pytorch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torchmetrics

import horovod.torch as hvd

from src.deep_learning.AMCDataset import AMCDataset
from src.deep_learning.ParamConfigurator import ParamConfigurator
from src.deep_learning.ClassificationTask import ClassificationTask
from src.deep_learning.ArchitectureSelector import ArchitectureSelector
from src.deep_learning.PerformanceAnalyst import PerformanceAnalyst

# def metric_average(val, name):
#     tensor = val.detach().clone()
#     avg_tensor = hvd.allreduce(tensor, name=name)
#     return avg_tensor.item()

# def test(model, test_sampler, test_loader):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#
#     model.eval()
#     test_loss = 0.
#     test_accuracy = 0.
#
#     for data, target in test_loader:
#         data, target = data.to(device), target.to(device)
#         output = model(data).clone().detach().cpu()
#         # sum up batch loss
#         test_loss += F.binary_cross_entropy_with_logits(output, target.clone().detach().cpu())
#         test_accuracy += torchmetrics.Accuracy()(output.round().int(), target.clone().detach().cpu().int())
#
#     # Horovod: use test_sampler to determine the number of examples in
#     # this worker's partition.
#     test_loss /= len(test_sampler)
#     test_accuracy /= len(test_sampler)
#
#     # Horovod: average metric values across workers.
#     test_loss = metric_average(test_loss, 'avg_loss')
#     test_accuracy = metric_average(test_accuracy, 'avg_accuracy')
#
#     # Horovod: print output only on first rank.
#     if hvd.rank() == 0:
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
#             test_loss, 100. * test_accuracy))


def main():

    hvd.init()

    # 1. Define configuration parameters
    config = ParamConfigurator()

    # 2. Select neural network architecture and create model
    selector = ArchitectureSelector(config=config)
    nn_model = selector.select_architecture()

    # 3. Define transformations
    transformations = transforms.Compose([transforms.ToTensor()])

    # 4. Initialize dataset
    dataset = AMCDataset(config, transform=transformations)

    # 5. Split dataset into train and val set
    train_data, val_data = random_split(dataset,
                                        [int(config.data_len*config.train_split),
                                         config.data_len - int(config.data_len*config.train_split)],
                                        generator=torch.Generator().manual_seed(42))

    # 6. Create dataloader
    train_data_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, **config.kwargs)
    validation_data_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, **config.kwargs)

    # test_sampler = torch.utils.data.distributed.DistributedSampler(
    #     val_data, num_replicas=hvd.size(), rank=hvd.rank())
    # test_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False,
    #                                           sampler=test_sampler, **config.kwargs)

    # 7. Start MLflow logging
    mlflow.set_tracking_uri(config.mlflow_log_dir)
    mlflow.set_experiment(config.experiment_name)
    # mlflow.pytorch.autolog()

    # 8. Create classifier
    classifier = ClassificationTask(nn_model=nn_model, config=config)

    # 9. Start training
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='',
        filename='model_parameters-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(max_epochs=config.num_epochs, deterministic=True, accelerator='horovod', gpus=1, precision=16, callbacks=[checkpoint_callback])
    # trainer = pl.Trainer(max_epochs=config.num_epochs, deterministic=True)
    trainer.fit(classifier, train_data_loader, validation_data_loader)

    if hvd.rank() == 0:
        mlflow.log_artifact(checkpoint_callback.best_model_path, artifact_path="best_model_params")
    classifier = classifier.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path, nn_model=nn_model, config=config, trained=True)

    # 10. Perform failure analysis
    analyst = PerformanceAnalyst(config=config, val_data=val_data, nn_model=classifier, trainer=trainer,
                                 val_dataloader=validation_data_loader)
    analyst.start_performance_analysis()

    # 11. Testing
    # test(nn_model, test_sampler, test_loader)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_deep_learning')

    main()

    logging.info('Finished main_deep_learning')
