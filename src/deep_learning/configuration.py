import torch

class training_configuration():
    batch_size = 4
    number_epochs = 50
    learning_rate = 0.0001
    resnet_depth = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    number_workers = 4
    plot_frequency = 10

class train_data_configuration():
    training_data_size = 10
    training_set_dir = "/workspace/dataset/voxel_data/"

class validation_data_configuration():
    validation_data_size = 10
    validation_set_dir = "/workspace/dataset/voxel_data/"
