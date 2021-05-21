import torch

def training_configuration():
    batch_size = 
    number_epochs = 
    learning_rate = 
    resnet_depth = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    number_workers = 4
    plot_frequency = 

def train_data_configuration():
    training_data_size = 
    training_set_dir = 
    training_gt_dir = 

def validation_data_configuration():
    validation_data_size = 
    validation_set_dir = 
    validation_gt_dir = 