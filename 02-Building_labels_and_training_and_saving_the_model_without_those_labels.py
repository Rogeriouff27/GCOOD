# This script performs the model training to extract the avgpool layer from the resnet and use this data to build the Voronoi diagram, and using graph coloring, the labels that will be removed in training and testing are calculated. Soon after, the training will be carried out without the chosen labels.

from models.model_util import pipeline_excl_lab_classifier
from const_voronoi.voronoi_utils import get_feat_vec_and_labels
from models.my_models import all_classifiers
from const_voronoi.config import config_voronoi
from const_voronoi.voronoi_utils import get_informatios_graph_coloring
from func_dataset.data_utils import return_labels_train_teste
from settings.config import config_data_classifiers #It will be used to configure the function return_labels_train_test function.
import torch
import pickle
import warnings
warnings.filterwarnings("ignore")
import os
import argparse

if __name__=='__main__':

    # Identifica o Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Path settings and the trained label
    parser = argparse.ArgumentParser(description = 'Taking the avgpoll layer from the Resnet classifier to build the Voronoi diagram.')

    # default variables
    dataset = 0   #0 for CIFAR10 and 1 for another dataset
    def_epoch = 20

    #Add arguments to argument list
    parser.add_argument('-d', '--dataset', type=int, default=dataset, help="Choose 0 for CIFAR10, or 1 for CIFAR100.")
    parser.add_argument('-e', '--epoch', type=int, default=def_epoch, help='Here the epochs are passed.')

    #Parse the arguments
    args = parser.parse_args()

    #Setting the number of epochs
    config_data_classifiers['epochs'] = args.epoch

    if(args.dataset == 0): 
        used_dataset = 'CIFAR10'
        num_class = 10
    elif(args.dataset == 1):
        used_dataset = 'CIFAR100'
        num_class = 100

    
    # Setting where model weights will be saved
    config_data_classifiers['path_folder_save_weight'] = os.path.join('data', used_dataset, 'classifiers', 'resnet', 'models', 'complete')
    #Configuration of the path where the data for the construction of the Voronoi diagram is stored
    config_voronoi['path_save_data_for_voronoi'] = os.path.join('data', used_dataset, 'data_for_const_voronoi')
    #Folder path where the label information that will be excluded in training and testing is saved.
    config_data_classifiers['path_dict_train_test'] = os.path.join('data', used_dataset, 'data_for_input')
    # used dataset
    config_data_classifiers['dataset'] = used_dataset
   

    #Conducting a training and testing pipeline with all classes
    pipeline_excl_lab_classifier(device, labels=['complete'])


    #Using the weights from the trained model
    model = all_classifiers(num_class = num_class , type_model = 'resnet', network_pretrained = False, path_weight = config_data_classifiers['path_folder_save_weight'], num_epoch = config_data_classifiers['epochs'])

    model.to(device)


    #Extracting data to build the Voronoi Diagram.
    feat_vec_data, labels_data = get_feat_vec_and_labels(device, model)


    # Saving the data to generate the Voronoi diagram.
    print('Saving the data to generate the Voronoi diagram...')

    arquivo = open('{}/{}.pck'.format(config_voronoi['path_save_data_for_voronoi'], 'feat_vec_data'), "wb")
    pickle.dump(feat_vec_data,arquivo)
    arquivo.close()

    arquivo = open('{}/{}.pck'.format(config_voronoi['path_save_data_for_voronoi'], 'labels_data'), "wb")
    pickle.dump(labels_data,arquivo)
    arquivo.close()

    # Extracting the label dictionary associated with the color chosen by the graph coloring algorithm.
    _, _, _, dic_label_color, _, _ = get_informatios_graph_coloring(config_data_classifiers['dataset'])


    #Returns the labels to be removed in training and testing, and saves this data.
    label_train, label_test, color_label = return_labels_train_teste(dic_label_color)

    print('label_train')
    print(label_train)
    print('\n')
    print('label_test')
    print(label_test)
    print('\n')
    print('color_label')
    print(color_label)

    # Labels that will be used to be excluded in training and testing
    exclusion_labels = []
    exclusion_labels.extend(label_train)
    exclusion_labels.extend(label_test)

    # Conducting the training without the chosen labels
    for labels in  exclusion_labels:
        pipeline_excl_lab_classifier(device, labels=labels)


