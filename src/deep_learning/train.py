import torch
import torchvision
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import wandb
from wandb_log import training_log
import configuration
from network.ResNet import ResNet


def wandb_initiliazer(arguments):
    with wandb.init(project="TUM_DI_Lab", config=arguments):
        config = wandb.config

        model, train_loader,val_loader,loss_fn,loss_fn1, optimizer = nn_model(config)

        train(model, train_loader, val_loader, loss_fn,loss_fn1, optimizer, config)
    return model


def nn_model(config):
    data_transforms = transforms.Compose([transforms.ToTensor()])


    train_set = ...
    validation_set = ...

    #Loading train and validation set
    train_set_loader = DataLoader(train_set,batch_size=config.batch_size,shuffle=False,num_workers=configuration.training_configuration.number_workers)
    validation_set_loader = DataLoader(validation_set,batch_size=config.batch_size,shuffle=False,num_workers=configuration.training_configuration.number_workers)

    #Build the model
    net = ResNet()

    if configuration.training_configuration.device.type == 'cuda':
        net.cuda()

   
    loss_function = torch.nn.BCELoss()
    

    optimizer = torch.optim.Adam(net.parameters(),lr=config.lr)
    
    return net,train_set_loader,validation_set_loader,loss_function,optimizer

def validation_phase(NN_model,val_set_loader,loss_function,epoch):
    print("Validating...")
    NN_model.eval()
    
    loss_val = 0

    for batch_id, (...) in enumerate(val_set_loader, 1):
        if(configuration.training_configuration.device.type == 'cuda'):
            pass

        output = NN_model(...)
        loss = loss_function(...)

        mini_batches += 1
        loss_val += float(loss)

    loss_val = loss_val/mini_batches
    return loss_val

def train(NN_model,train_set_loader,val_set_loader,loss_function,optimizer, config):
    wandb.watch(NN_model,loss_function,log='all',log_freq=50)

    num_seen_samples = 0
    mini_batches = 0
    loss_value = 0

    for epoch in range(config.epochs):
        for batch_id, (...) in enumerate(train_set_loader, 1):
            NN_model.train()
            if(configuration.training_configuration.device.type == 'cuda'):
                pass


            output = NN_model(...)
            loss = loss_function(...)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_seen_samples += len(...)
            mini_batches += 1
            loss_value += float(loss)

            if (mini_batches % 200) == 0:
                print("Training loss after %d batches"%(int(num_seen_samples/configuration.training_configuration.train_batch_size)))


            #Plotting in wandb
            if (mini_batches % configuration.training_configuration.plot_frequency == 0):
                val_loss = validation_phase(NN_model,val_set_loader,loss_function,epoch)
                training_log(loss_value/configuration.training_configuration.plot_frequency, mini_batches)
                training_log(val_loss,mini_batches,False)

                PATH = "model.pt"
                torch.save({'epoch':epoch,'model_state_dict':NN_model.state_dict(),'optimimizer_state_dict':optimizer.state_dict(),'loss':loss_value},PATH)

                loss_value = 0

            print('Epoch-{0} lr: {1:f}'.format(epoch, optimizer.param_groups[0]['lr']))

def training_log(loss,mini_batch,train=True):
    if train == True:
        wandb.log({'batch':mini_batch,'loss':loss})
    else:
        wandb.log({'batch':mini_batch,'loss':loss})