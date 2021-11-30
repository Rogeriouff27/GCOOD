# Save pre-processed training and test data, which will be used by the GCOOD detector.

import torch
from binary.binary_utils import get_trein_test_save
from settings.config import config_data_classifiers
import os
import argparse


if __name__=='__main__':
    
    #Path settings and the trained label
    parser = argparse.ArgumentParser(description = 'This variable will help configure the get_train_test_save function.')

    # default variables
    dataset = 0   #0 for CIFAR10 and 1 for CIFAR100

    #Add arguments to argument list
    parser.add_argument('-d', '--dataset', type=int, default=dataset, help="Choose 0 for CIFAR10, or 1 for CIFAR100.")

    #Parse the arguments
    args = parser.parse_args()
    
    if(args.dataset == 0): 
        used_dataset = 'CIFAR10'
    elif(args.dataset == 1):
        used_dataset = 'CIFAR100'
        

    #Settings
    config_data_classifiers['path_dict_train_test'] = os.path.join('data', used_dataset, 'data_for_input')



    # Identifica o Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_lab_bin, valid_lab_bin, test_lab_bin, label_train, label_test, color_label = get_trein_test_save(device)
