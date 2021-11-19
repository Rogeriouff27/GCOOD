from models.my_models import GCOOD_network
from settings.config import config_GCOOD_network
from models.model_util import GCOOD_pipeline
import torch
import torch.nn as nn
import os
import argparse

if __name__=='__main__':

    #Path settings and the trained label
    parser = argparse.ArgumentParser(description = 'This path will be used to store the GCOOD model.')

    # default variables
    dataset = 0   #0 for CIFAR10 and 1 for another dataset
    def_epoch = 150
    #def_path_root = os.path.join('data', 'cifar10', 'GCOOD_network', 'model')

    #Add arguments to argument list
    parser.add_argument('-d', '--dataset', type=int, default=dataset, help="Choose 0 for cifar10, or 1 for another dataset.")
    parser.add_argument('-e', '--epoch', type=int, default=def_epoch, help='Here the epochs are passed.')

    #Parse the arguments
    args = parser.parse_args()
    
    if(args.dataset == 0): 
        used_dataset = 'cifar10'
    elif(args.dataset == 1):
        used_dataset = 'another_dataset'

    #Settings
    config_GCOOD_network['path_folder_save_weight'] = os.path.join('data', used_dataset, 'GCOOD_network', 'model')
    config_GCOOD_network['epochs'] = args.epoch

    # Identify the Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #Makes the mean of the weights equal to 0 and the standard deviation equal to 0.02.
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    model = GCOOD_network(10,10,config_GCOOD_network)
    model.to(device)


    model.apply(weights_init)

    # Train and test the net, and save the weights.
    GCOOD_pipeline(device, model)