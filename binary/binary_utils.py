from func_dataset.data_utils import loady_dataset, format_dataset_for_GCOOD, CifarDataset
from func_dataset.data_utils import pre_process_dataset_for_GCOOD, augmentation_data_for_GCOOD
from settings.config import config_data_classifiers
from settings.config import config_GCOOD_network
from func_dataset.data_utils import return_labels_train_teste
from func_dataset.data_utils import buid_dataset, buid_all_dataset
import torch
import pickle


# This function only returns 0 if the target passed is in label, or 1 if not.
def return_class_binary(labels, target):
    if(target in labels):
        return 0
    else:
        return 1

    
# It takes the training data from cifar10 through the loady_dataset function, transforms it as label 0 if it belongs to list_label or 1 otherwise, and puts this information in dictionary form. The list_labels argument will be a list containing lists of labels
def get_format_dataset_input(device, list_labels, mode_data = 'train'):
    
    # Data reading
    _, test_data = loady_dataset()
    
    
    # If the data is from training 
    if(mode_data == 'train'):
        data1,_,_ = pre_process_dataset_for_GCOOD(test_data)
        data_change = augmentation_data_for_GCOOD(data1, num_copy = 1)
        pre_process_data = CifarDataset(data_change)
        data = buid_all_dataset(all_dataset = pre_process_data, batch_size = 1000)
        
    # If the data is for validation
    elif(mode_data == 'valid'):
        _, data1,_ = pre_process_dataset_for_GCOOD(test_data)
        pre_process_data = CifarDataset(data1)
        data = buid_all_dataset(all_dataset = pre_process_data, batch_size = 1000)
      
    # If the data is test
    elif(mode_data == 'test'):
        _, _, data1 = pre_process_dataset_for_GCOOD(test_data)
        pre_process_data = CifarDataset(data1)
        data = buid_all_dataset(all_dataset = pre_process_data, batch_size = 1000)
        
    else:
        raise ValueError(f'Invalid mode_data({mode_data})')
    
    # Data dictionary
    dic_dataset = {}
    dic_dataset['data'] = []
    dic_dataset['targets'] = []
    
    # It will go through all the possible labels and remove all outputs for each of these labels
    size_list = len(list_labels)
    for i in range(size_list):
        
        dic_train = format_dataset_for_GCOOD(device, data, list_labels[i], 20)
        dic_dataset['data'].extend(dic_train['data'])
        
        dic_train['targets'] = [return_class_binary(list_labels[i], x) for x in dic_train['targets']]
        dic_dataset['targets'].extend(dic_train['targets'])
        
    dic_dataset['size_data'] = len(dic_dataset['targets'])

    
    return dic_dataset


# This function extracts the training and test data using the training and test labels and saves this data.
def get_trein_test_save(device):
    
    # Opening training and test labels files
    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'label_train'), "rb")
    label_train = pickle.load(arquivo)
    arquivo.close()

    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'label_test'), "rb")
    label_test = pickle.load(arquivo)
    arquivo.close()
    
    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'color_label'), "rb")
    color_label = pickle.load(arquivo)
    arquivo.close()

    
    # Taking training and testing data
    train_lab_bin = get_format_dataset_input(device, label_train, mode_data = 'train')
    valid_lab_bin = get_format_dataset_input(device, label_test, mode_data = 'valid')
    test_lab_bin = get_format_dataset_input(device, label_test, mode_data = 'test')
    
    # Saving training and testing datasets
    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'train_lab_bin'), "wb")
    pickle.dump(train_lab_bin,arquivo)
    arquivo.close()
    
    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'valid_lab_bin'), "wb")
    pickle.dump(valid_lab_bin,arquivo)
    arquivo.close()

    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'test_lab_bin'), "wb")
    pickle.dump(test_lab_bin,arquivo)
    arquivo.close()
    
    return train_lab_bin, valid_lab_bin, test_lab_bin, label_train, label_test, color_label

# Class to prepare the dataset
class DatasetLabelBinary(torch.utils.data.Dataset):
    
    # Constructor method
    def __init__(self, dataset):
        
        # Dataset
        self.image = dataset['data']
        self.targets = dataset['targets']
       
    # Method for obtaining image indexes
    def __getitem__(self, idx):
        
        # The image and target that are being worked on
        image = self.image[idx]
        targets = self.targets[idx]
        image = torch.tensor(image)
        targets = torch.tensor(targets)

        return image, targets

    # Method for calculating dataset size
    def __len__(self):
        return len(self.targets)
    
    
# This function takes the pre-processed data from the output of the resnet152 network, processes this data and returns it ready to be used on the network.
def get_train_test_for_network():
    
    # Opening files
    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'train_lab_bin'), "rb")
    train_lab_bin = pickle.load(arquivo)
    arquivo.close()
    
    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'valid_lab_bin'), "rb")
    valid_lab_bin = pickle.load(arquivo)
    arquivo.close()

    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'test_lab_bin'), "rb")
    test_lab_bin = pickle.load(arquivo)
    arquivo.close()
    
    # Pre-processing the data
    train_data = DatasetLabelBinary(train_lab_bin)
    valid_data = DatasetLabelBinary(valid_lab_bin)
    test_data = DatasetLabelBinary(test_lab_bin)
    
    
    train_loader = buid_all_dataset(train_data, config_GCOOD_network['batch_size'])
    valid_loader = buid_all_dataset(valid_data, config_GCOOD_network['batch_size'])
    test_loader = buid_all_dataset(test_data, config_GCOOD_network['batch_size'])
    
    return train_loader, valid_loader, test_loader