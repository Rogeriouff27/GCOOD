from DCGAN.config import config_gan
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn.parallel
import torch.backends.cudnn as cudnn


#This function serves to get the data from CIFAR10.
# The normalization was taken from the website: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def loady_dataset():
    
    # Transformations for Input Data.
    transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
    # Load Training and Test Data.
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

# This function receives as input data from the get_dic_dataset function, which passes a dictionary with the information main dataset. This function aims to collect only information from targets that are not in labels.
def get_dic_label_del(data, labels):
    
    # Mapping the old label number to the new one. The first column is the old label and the second is the new one or -1 if it has been 
    # deleted.
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
            
        # Building the dictionary by removing the informed labels and their associated information.
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


# This class receives data from the get_dic_label_del function. This class is just a pre-processing of the data to be passed to the 
# buid_dataset function.
class PreProcessDataset(torch.utils.data.Dataset):
    
    # Método construtor
    def __init__(self, dataset):
        
        # Dataset
        self.image = dataset['data']
        self.targets = dataset['targets']
       
    # Method for obtaining image indexes.
    def __getitem__(self, idx):
        
        # The image and target that are being worked on.
        image = self.image[idx]
        targets = self.targets[idx]
        
        transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        
        image = transformations(image)
        
        targets = torch.tensor(targets)

        return image, targets
    
     # Method for calculating dataset size.
    def __len__(self):
        return len(self.targets)
    
    
# This function takes the data from the PreProcessDataset and returns the data prepared for use with the model.
def buid_dataset(all_dataset, batch_size):
    
        
    # Size of train_data that will be used to create train and val indices.
    training_size = len(all_dataset.targets)
    
    # Index used to get train and val data.
    training_indices = list(range(training_size))
    np.random.shuffle(training_indices)
    
    # Training and Validation Samples.
    training_sample = SubsetRandomSampler(training_indices)
    
    # Generate the Final Data Samples.
    all_dataset_loader = DataLoader(all_dataset, batch_size = batch_size, sampler = training_sample)
    
    return all_dataset_loader

# This function receives as parameter data coming from the loady_dataset function and labels from which the data will be deleted and returns the data only with the information of the labels that are not contained in the passed labels parameter. This data is ready to be used by the model.
def get_dataset_for_gans(dataset, labels, batch_size):
    
    # Input dataset.
    data = get_dic_dataset(dataset)
    all_data = get_dic_label_del(data, labels)
    
    # Number of classes.
    num_class = all_data['num_class']
    
    
    # Building the output dataset.
    all_data = PreProcessDataset(all_data)
    all_data_loader = buid_dataset(all_data, batch_size)
    
    # Checking output data.
    assert list(next(iter(all_data_loader))[0][0].shape) == [config_gan['nc'], 32, 32], 'Expected output has the format [size_bath,{},{},{}]'.format(3, 32, 32)
    
    return all_data_loader

# Plot some data grid.
def plotting_grid_some_images(device, dataset, message = '', num_line = 8, num_colum = 8, fig_size = 8):
    
    # Plot some training images.
    real_batch = next(iter(dataset))
    plt.figure(figsize=(fig_size,fig_size))
    plt.axis("off")
    plt.title(message)
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:num_line*num_colum], nrow= num_colum, padding=2, normalize=True).cpu(),(1,2,0)))
    
    
# This function takes an image in float format and converts it to uint8 format.
def normalize8(I):
    
    mn = I.min()
    mx = I.max()
    
    mx -= mn
    I = ((I - mn)/mx) * 255
    
    return I.astype(np.uint8)

# This function receives a batch of images in torch float format and converts them to numpy uint8 already labeled with the label.
def convert_for_numpy_uint8_with_label(data, label):
    
    batch_size = data.shape[0]
    data = np.array(data)
    dataset = {}
    dataset['data'] = []
    dataset['targets'] = []
    for i in range(batch_size):
        dataset['data'].append(normalize8(np.transpose(data[i],(1,2,0))))
        dataset['targets'].append(label[0])
    return dataset