#This script will be used to create fake data for each class. Therefore, it must be run for each data belonging to a certain class that you want to generate.

from DCGAN.config import config_gan
from DCGAN.model_dcgan import pipeline_dcgan
from DCGAN.model_dcgan import plotting_animation_training
from func_dataset.data_utils import loady_dataset
import torch
import warnings
warnings.filterwarnings("ignore")
import os
import argparse

if __name__ == '__main__':
    
    # Identify the Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path settings and the trained label
    parser = argparse.ArgumentParser(description = 'Path settings and the trained label')

    # default variables
    dataset = 0   #0 for CIFAR10 and 1 for another dataset
    def_epoch = 45000

    # Add arguments to argument list
    parser.add_argument('-d', '--dataset', default=dataset, type=int, help="Choose 0 for CIFAR10, 1 CIFAR100.")
    parser.add_argument('-e', '--epoch', type=int, default=def_epoch, help='Here the epochs are passed. Trainings are saved, in epochs multiple of 100.')

    # Parse the arguments
    args = parser.parse_args()

    # Setting the number of epochs
    config_gan['epochs'] = args.epoch

    # Choosing the used dataset
    if(args.dataset == 0):
        # Configuration of the path where the folder is located
        config_gan['path_weight'] = os.path.join('data', 'CIFAR10' , 'gans_model')
        config_gan['dataset'] = 'CIFAR10'
        all_labels = list(range(10))
    elif(args.dataset == 1):
        # Configuration of the path where the folder is located
        config_gan['path_weight'] = os.path.join('data', 'CIFAR100', 'gans_model')
        config_gan['dataset'] = 'CIFAR100'
        all_labels = list(range(100))


    #Pipeline gan training BCGAN iteratively going through all classes
    for label in all_labels:
        labels_for_training = [label]
        netG, netD, img_list, G_losses, D_losses = pipeline_dcgan(device, labels_for_training)



