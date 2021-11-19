from DCGAN.model_dcgan import get_dataset_fake_configured  # used in get_train_test_excl_lab function
from settings.config import config_data_classifiers, config_GCOOD_network
from models.my_models import all_classifiers
from torchvision import datasets
from torchvision import transforms
import torchvision
import numpy as np
import random
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import albumentations
import albumentations.pytorch
import pickle

#This function serves to get the data from CIFAR10
# The transformation was taken from website reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def loady_dataset():
    
    # Transformations for Input Data
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load Training and Test Data
    train_data = datasets.CIFAR10('CIFAR10', train = True, download = True, transform = transformations)
    test_data = datasets.CIFAR10('CIFAR10', train = False, download = True, transform = transformations)
    
    return train_data, test_data


# This function takes the main information from the CIFAR10 dataset like: class, images, targets, the dataset size and the number of classes that is given by the loady_dataset function and returns a dictionary with this information.
def get_dic_dataset(data):
  
    my_data = {}
    my_data['class'] = data.classes
    my_data['data'] = data.data
    my_data['targets'] = data.targets
    my_data['size_data'] = len(data.targets)
    my_data['num_class'] = len(data.classes)
    
    return my_data


# This function is intended to increase the amount of data. The train_data, test_data data are data obtained from the loady_dataset function and the num_copy argument of this function is used to say, how many times you want your dataset to be increased. In this case, it was chosen as the default that the dadaset was increased by 10 times. The dataset argument is to ensure that when the dataset is cifar10, the settings for this data will be correct for both input and output data. The function also has the option to add fake images generated by a gan, which is by default use_gan=False.
def augmentantion_dataset(train_data, test_data, use_gan = False, data_gan = None, num_copy = 10):
    
    # Taking essential training and testing information and putting it in a dictionary
    train =  get_dic_dataset(train_data)
    test =  get_dic_dataset(test_data)
    
    # Increasing the data
    data = []
    for _ in range(num_copy):
        temp = [train['data'][i] for i in range(train['size_data'])]
        data.extend(temp)
        
    targets = []
    for _ in range(num_copy):
        temp = [train['targets'][i] for i in range(train['size_data'])]
        targets.extend(temp)  
        
    if(use_gan):
        
        assert data_gan != None, 'With use_gan = True, it is mandatory to pass the dataset generated by the gan'
        
        temp = [data_gan['data'][i] for i in range(len(data_gan['targets']))]
        data.extend(temp)
        
        temp = [data_gan['targets'][i] for i in range(len(data_gan['targets']))]
        targets.extend(temp)
        
    data = np.array(data)

    train['data'] = data
    train['targets'] = targets
    train['size_data'] = len(train['targets'])
    
    return train, test


# This function receives as input data from the get_dic_dataset function, which passes a dictionary with the information main dataset. This function aims to collect only information from targets that are not in labels.
def get_dic_label_del(data, labels):
    
    # Mapping the old label number to the new one. The first column is the old label and the second is the new one or -1 if it has been deleted
    if(labels[0] == 'complete'):
        data_change = {}
        data_change['class'] = data['class'].copy()
        data_change['data'] = data['data'].copy()
        data_change['targets'] = data['targets'].copy()
        data_change['size_data'] = len(data['targets'])
        data_change['num_class'] = len(data_change['class'])
        
    else:
        count = 0
        map_labels = []
        for i in range(data['num_class']):
            if i not in labels:
                map_labels.append([i,i-count])
            else:
                map_labels.append([i,-1])  
                count += 1
            
        # Building the dictionary by removing the informed labels and their associated information
        data_change = {}
        data_change['class'] = [data['class'][i] for i in range(data['num_class']) if i not in labels]
        data_change['data'] = []
        data_change['targets'] = []

        for i in range(data['size_data']):
            if(map_labels[data['targets'][i]][1] != -1):
                data_change['data'].append(data['data'][i])
                data_change['targets'].append(map_labels[data['targets'][i]][1])
        
        data_change['data'] = np.array(data_change['data']) 
        data_change['size_data'] = len(data_change['targets'])
        data_change['num_class'] = data['num_class'] - len(labels)
    
    return data_change


# This function is just used to avoid calling the get_dic_label_del function twice to get training and testing, so with just this function we can get this data. 
def get_train_test_label_del(train_data, test_data, labels):
     
    # Building the training and test datasets removing the informed labels and their information
    data_train = get_dic_label_del(train_data, labels)
    data_test = get_dic_label_del(test_data, labels)
    
    return data_train, data_test


# This class receives data from the get_dic_label_del function. This class is just a pre-processing of the data to be passed to the buid_dataset function.
class CifarDataset(torch.utils.data.Dataset):
    
    # Constructor method
    def __init__(self, dataset, transform=None):
        
        # Dataset
        self.image = dataset['data']
        self.targets = dataset['targets']
        self.transform = transform
       
    # Method for obtaining image indexes
    def __getitem__(self, idx):
        
        # The image and target that are being worked on
        image = self.image[idx]
        targets = self.targets[idx]
        
        if self.transform:
            augmented = self.transform(image=image) 
            image = augmented['image']
        else:
            trans = albumentations.Compose([
            albumentations.pytorch.transforms.ToTensor()
            ])
            augmented = trans(image=image)
            image = augmented['image']
        
        targets = torch.tensor(targets)

        return image, targets
    
     # Method for calculating dataset size
    def __len__(self):
        return len(self.targets)


# This function receives the data from CifarDataset and returns the data prepared to be used with the model.
def buid_dataset(train_data, test_data, batch_size, validation_size = 0.2):
    
        
    # Size of train_data that will be used to create train and val indices
    training_size = len(train_data.targets)
    
    # Index used to get train and val data
    indices = list(range(training_size))
    np.random.shuffle(indices)
    index_split = int(np.floor(training_size * validation_size))
    
    # Indices for Training and Validation Data
    validation_indices, training_indices = indices[:index_split], indices[index_split:]
    
    # Training and Validation Samples
    training_sample = SubsetRandomSampler(training_indices)
    validation_sample = SubsetRandomSampler(validation_indices)
    
    # Generate the Final Data Samples
    train_loader = DataLoader(train_data, batch_size = batch_size, sampler = training_sample)
    valid_loader = DataLoader(train_data, batch_size = batch_size, sampler = validation_sample)
    test_loader = DataLoader(test_data, batch_size = batch_size)
    
    return train_loader, valid_loader, test_loader


# This function processes the dataset and returns it complete without breaking it into validation data..
def buid_all_dataset(all_dataset, batch_size):
    
        
    # Size of train_data that will be used to create train indices
    training_size = len(all_dataset.targets)
    
    # Index used to get train data 
    training_indices = list(range(training_size))
    np.random.shuffle(training_indices)
    
    # Training Samples
    training_sample = SubsetRandomSampler(training_indices)
    
    # Generate the Final Data Samples
    all_dataset_loader = DataLoader(all_dataset, batch_size = batch_size, sampler = training_sample)
    
    return all_dataset_loader

# Transformation is faster than torchvision. Its documentation can be found on the website:
# https://albumentations.ai/docs/
# This normalization of the images is equivalent to the one taken from the website:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
albumentations_transform_oneof = albumentations.Compose([
    albumentations.OneOf([
                          albumentations.HorizontalFlip(p=1),
                          albumentations.RandomRotate90(p=1),
                          albumentations.Rotate(30, p=1),
                          albumentations.Rotate(50, p=1),
                          albumentations.Rotate(70, p=1),
                          albumentations.VerticalFlip(p=1)            
    ], p=0.5),
    albumentations.OneOf([
                          albumentations.MotionBlur(p=1),
                          albumentations.OpticalDistortion(p=1),
                          albumentations.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
                          albumentations.GaussNoise(p=1)                 
    ], p=0.5),
    albumentations.pytorch.ToTensor()
])



# This function receives a list of labels from which data referring to these labels should be deleted, fake data obtained by gan is also added, and an increase in data.
def get_train_test_excl_lab(device, labels, batch_size):
    
    # Input dataset
    train_data, test_data = loady_dataset()
    dataset_fake = get_dataset_fake_configured(device)
    train_data, test_data = augmentantion_dataset(train_data, test_data, use_gan = True, data_gan = dataset_fake)
    train_changed, test_changed = get_train_test_label_del(train_data, test_data, labels)
    
    # Number of classes
    num_class = train_changed['num_class']
    
    
    # Building the output dataset
    train_changed = CifarDataset(train_changed, albumentations_transform_oneof)
    test_changed = CifarDataset(test_changed)
    train_loader, valid_loader, test_loader = buid_dataset(train_changed, test_changed, batch_size)
    
    return train_loader, valid_loader, test_loader, num_class

# Passing a list as a set, this function calculates all non-empty subsets of this list.
def calculates_subsets(s):
    x = len(s)
    subset = []
    for i in range(1,1 << x):
        subset.append([s[j] for j in range(x) if (i & (1 << j))])
    return subset


# This function builds the training and test labels and a dictionary that associates color with their labels.
def return_labels_train_teste(dic_label_color):
    
    # Building the dictionary where the key is the color and the value are the labels associated with this color
    keys = sorted(dic_label_color)
    color_label = {}
    for key in keys:
        list_label = sorted([x for x,y in zip(dic_label_color.keys(),dic_label_color.values()) if y == key])
        if(list_label):
            color_label[key] = list_label
    color_label['without_color'] = ['complete']
    
    # For each value of the color_label dictionary this list returns the number of non-empty subsets
    num_subsets_each_dic_value = [2**len(x)-1 for x in color_label.values()]
    
    # Calculating the size of the list that will be split for training and testing
    size_list_label = sum(num_subsets_each_dic_value)
    
    # Calculating the probability that each color_label key will be chosen as a function of the size of 
    # subsets that the value of that key generates. The last color_label key will not be considered as 
    # this key is 'without_color' where its value is 'complete' which represents a set that no labels 
    # will be removed.
    weights = np.array(num_subsets_each_dic_value[0:(len(num_subsets_each_dic_value)-1)])*100/size_list_label
    weights = list(np.array(weights, dtype = np.int16))
    
    # Training and Testing Label
    label_test = [['complete']]
    label_train = []
    
    # Each element of keys will be chosen according to the probability weights that will be used to fill
    # label_test
    keys = list(color_label.keys())
    keys = keys[0:len(keys)-1]
    
    # Calculating the size of label_test and label_train
    size_test = int(size_list_label/3)
    size_train = size_list_label - size_test
    
    # Filling in label_test and label_train. First, label_test is filled using probability weights for 
    # choosing keys so that the size of label_test is equal to size_test and the remaining keys are used to build label_train.
    while(size_test> 1):
        key = random.choices(keys, weights, k=1)[0]
        if(num_subsets_each_dic_value[key] < size_test):
            label_test.extend(calculates_subsets(color_label[key]))
            size_test -= num_subsets_each_dic_value[key]
            index = keys.index(key)
            del(keys[index])
            del(weights[index]) 
    for key in keys:
        label_train.extend(calculates_subsets(color_label[key]))
        
        
    # The 'complete' training option will be excluded.
    del label_test[0]
        
    # Saving training, test and color label files
    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'label_train'), "wb")
    pickle.dump(label_train,arquivo)
    arquivo.close()

    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'label_test'), "wb")
    pickle.dump(label_test,arquivo)
    arquivo.close()

    arquivo = open('{}/{}.pck'.format(config_data_classifiers['path_dict_train_test'], 'color_label'), "wb")
    pickle.dump(color_label,arquivo)
    arquivo.close()
        
    return label_train, label_test, color_label


# This function takes the dataset generated by the get_dic_dataset function and splits it into two training and testing datasets in dictionary format.
def split_train_test(dataset, cut_data_for_train = 0.7):
    
    # Calculating the dataset size that each training dataset label will have
    size_train = int(dataset['size_data']*cut_data_for_train*(1/10))
        
    # dados de treino
    data_train = {}
    data_train['class'] = dataset['class']
    data_train['data'] = []
    data_train['targets'] = []
    data_train['num_class'] = dataset['num_class']

    # dados de teste
    data_test = {}
    data_test['class'] = dataset['class']
    data_test['data'] = []
    data_test['targets'] = []
    data_test['num_class'] = dataset['num_class']


    list_cout = []
    for i in range(dataset['num_class']):
        list_cout.append(0)

    for i in range(dataset['size_data']):
    
        if(list_cout[dataset['targets'][i]] < size_train):
            data_train['data'].append(dataset['data'][i])
            data_train['targets'].append(dataset['targets'][i])
            list_cout[dataset['targets'][i]] += 1
        else:
            data_test['data'].append(dataset['data'][i])
            data_test['targets'].append(dataset['targets'][i])
        
    data_train['data'] = np.array(data_train['data'])
    data_test['data'] = np.array(data_test['data'])

    data_train['data'], data_train['targets'] = shuffle(data_train['data'], data_train['targets'])
    data_test['data'], data_test['targets'] = shuffle(data_test['data'], data_test['targets'])

    data_train['size_data'] = len(data_train['targets'])

    data_test['size_data'] = len(data_test['targets'])

    for i in range(dataset['num_class']):
        assert data_train['targets'].count(i) == size_train, 'data_train["targets"].count({} = {})'.format(i,size_train)
    
    for i in range(dataset['num_class']):
        assert data_test['targets'].count(i) == int((dataset['size_data'] - dataset['num_class']*size_train)/dataset['num_class']), 'data_test["targets"].count({} = {})'.format(i,dataset['size_data'] - int((dataset['size_data'] - dataset['num_class']*size_train)/dataset['num_class']))
    
    return data_train, data_test

# Increase second network dataset
def augmentation_data_for_GCOOD(dataset, num_copy = 1):
    
    # Increasing the data
    data = []
    for _ in range(num_copy):
        temp = [dataset['data'][i] for i in range(dataset['size_data'])]
        data.extend(temp)
        
    targets = []
    for _ in range(num_copy):
        temp = [dataset['targets'][i] for i in range(dataset['size_data'])]
        targets.extend(temp)  
        
    data = np.array(data)

    dataset['data'] = data
    dataset['targets'] = targets
    dataset['size_data'] = len(dataset['targets'])
    
    return dataset

# This function takes the test data from the cifar10 function, divides it into training and testing and validation increases the training data, and returns it already formatted to be used by the network.
# The dataset will come from loady_dataset and divide by 60% for training 20% for validation and testing.
def pre_process_dataset_for_GCOOD(dataset, cut_data_for_train = 0.6, num_copy_aug_train = 1):
    
    # Extracting the dataset
    dataset = get_dic_dataset(dataset)
    train_data, test = split_train_test(dataset, cut_data_for_train)
    test_data, valid_data = split_train_test(test, 0.5)
    
    return train_data, valid_data, test_data


# This function builds a matrix with 10 rows and (10 - labels size) columns that will be used to make a transformation.
# For a better understanding of this transformation, let's take an example. Suppose the passed label is [0, 5, 6], then we have that these labels are considered without labels, if we have a vector of the form [u1,u2,u3,u4,u7,u8,u9], applying the transformation on this vector we want it to output as [0,u1,u2,u3,u4,0,0,u7,u8,u9].
def transform_linear_data(label):
    if(label[0] == 'complete'):
        return np.identity(10, dtype =np.float32)
    else:
        dim_row = 10
        dim_column = 10 - len(label)
        linear_transformation = np.zeros([dim_row, dim_column], dtype=np.float32)
        count = 0
        for i in range(10):
            if(i not in label):
                linear_transformation[i][count] = 1
                count += 1
            else:
                try:
                    linear_transformation[i][count] = 0
                except IndexError:
                    linear_transformation[i][dim_column-1] = 0
                
        return linear_transformation
    
    
# This function takes the training dataset from cifar10, which was split is also split again into training and testing. It takes one of these datasets, passes it to the first network trained with the weights chosen according to the label passed, takes this result and the targets and puts it in a dictionary to be used by yet another function, until it is used by the next network. Note, the dataset_loader has all classes, and classes according to rank will be labeled later as 0 or 1.
def format_dataset_for_GCOOD(device, dataset_loader, labels, num_epoch):

    
    # Extracting the number of classes to be passed to the network
    if(labels[0] == 'complete'):
        num_class = 10
    else:
        num_class = 10 - len(labels)

    # Path where the weights of the trained model are saved
    root = config_data_classifiers['path_folder_save_weight'].split('models')[0] + 'models/'
    path_weight = root + '_'.join(map(str, labels))
    
    # Model used.
    # Note that we are taking the data with all the classes, but going through a network that has 
    # learned from the classes minus the classes that belong to the labels set passed to this function.
    model = all_classifiers(num_class, config_GCOOD_network['model'], False, path_weight, num_epoch)
    model.eval()
    model.to(device)
    
    # The softmax will be applied to each output with the intention that each output is greater than or equal to zero.
    m = nn.Softmax()
    
    # Applying the model and calculating training and testing outputs
    dic_train = {} # Training Data Dictionary
    dic_train['data'] = []
    dic_train['targets'] = []
    for data, target in dataset_loader:
        data, target = data.to(device), target.to(device)
        # Passing the softmax for the values to be greater than zero then put it in cpu and convert it to numpy
        output = m(model(data)).cpu().detach().numpy() 
        dic_train['data'].extend(output)
        dic_train['targets'].extend(target.cpu().tolist())   
      
        
    # Applying transforms to each output so that each output contains 10 columns.
    # The transformation fills the labels that were deleted by zero.
    trans_lin_data = transform_linear_data(labels)
    
    dic_train['data'] = [np.sort(np.dot(trans_lin_data, x)) for x in dic_train['data']]
    
    return dic_train 


# Takes the training and test dataset from cifar10, and performs the union.
def get_union_dataload_cifar10(batch_size):
    
    train_data, test_data = loady_dataset()
    train = get_dic_dataset(train_data)
    test = get_dic_dataset(test_data)

    # União dos dadasets
    data_union = {}
    data_union['class'] = train['class']

    data_train = [x for x in train['data']]
    data_test = [x for x in test['data']]
    data_union['data'] = []
    data_union['data'].extend(data_train)
    data_union['data'].extend(data_test)
    data_union['data'] = np.array(data_union['data'])

    data_train_targets = [x for x in train['targets']]
    data_test_targets = [x for x in test['targets']]
    data_union['targets'] = []
    data_union['targets'].extend(data_train_targets)
    data_union['targets'].extend(data_test_targets)

    data_union['size_data'] = len(data_union['targets'])
    data_union['num_class'] = len(data_union['class'])
    
    
    change_data_union = CifarDataset(data_union)
    data_union = buid_all_dataset(change_data_union, batch_size)
    
    return data_union