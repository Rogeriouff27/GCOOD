from func_dataset.data_utils import loady_dataset, build_dataset
from settings.config import config_data_classifiers
import torch
from const_voronoi.config import config_voronoi
import pickle
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial import Voronoi
import networkx as nx
import networkx.drawing
from matplotlib import pyplot as plt
import os


# This function is used to get the avgpool layer, which is used in the get_data_for_const_voronoi function.
my_output = None
def my_hook(self, input, output):
    global my_output 
    my_output = output


# This function aims to capture the data coming out of the model's avgpool layer with training and testing data which will be grouped, to be used in constructing the Voronoi diagram.
def get_feat_vec_and_labels(device, model):
    
    # Using the hook to get the model's avgpool layer
    model.avgpool.register_forward_hook(my_hook)
    
    # extracting the dataset
    train_data, test_data = loady_dataset()
    train_loader, _, test_loader = build_dataset(train_data, test_data, 1000, validation_size = 0)
    
    feat_vec_data = []
    labels_data = []
    # List that will capture feature_vector data and their respective training and test labels
    for data, label in  train_loader:
        data, label = data.to(device), label.to(device)
        model(data)
        feat_vec_data.extend(torch.squeeze(my_output).cpu().tolist())
        labels_data.extend(label.cpu().tolist())
    
    for data, label in  test_loader:
        data, label = data.to(device), label.to(device)
        model(data)
        feat_vec_data.extend(torch.squeeze(my_output).cpu().tolist())
        labels_data.extend(label.cpu().tolist())
    
    return feat_vec_data, labels_data

# This function sorts the list of feat_vec output from the get_data_for_const_voronoi function and organizes this list by labels, that is, feat_vec with label 0 appear first, then feat_vec with label 1, then label2, and so on to the last label.
def sort_feature_using_labels(feat_vec, labels, num_class):
    
    # First column is the label and second is the order in which it appears in the list
    order = [(y,x) for x, y in enumerate(labels) ]
    order = sorted(order)
    # feat_vec_ordered_by_labels is a list of feat_vec ordered by label, from label 0 to last label
    feat_vec_ordered_by_labels = []
    # break_feat these are the divisions where one class ends and another starts
    break_feat = [0]
    for k in range(num_class):
        temp = []
        break_lab = None
        for count,(i,j) in enumerate(order):
            if i == k:
                temp.append(feat_vec[j])
                break_lab = count 
        break_feat.append(break_lab+1)
        feat_vec_ordered_by_labels.extend(temp)
    return feat_vec_ordered_by_labels, break_feat


# This function takes data from the resnet's avgpool layer and transforms it into data to be used in building the Voronoi diagram.
def get_data_for_voronoi(num_comp_pca):
    
    # File path
    path = "/"+ os.path.join('root', 'PROJETOS', 'GCOOD', 'data', config_data_classifiers['dataset'], 'data_for_const_voronoi')
    config_voronoi['path_save_data_for_voronoi'] = path
    
    # Opening files
    arquivo = open('{}/{}.pck'.format(config_voronoi['path_save_data_for_voronoi'], 'feat_vec_data'), "rb")
    feat_vec_data = pickle.load(arquivo)
    arquivo.close()

    arquivo = open('{}/{}.pck'.format(config_voronoi['path_save_data_for_voronoi'], 'labels_data'), "rb")
    labels_data = pickle.load(arquivo)
    arquivo.close()
    
    # Ordering feat_vec_ordered_by_labels according to their labels, that is, first come the vectors from 
    # label 0, after label 1, and proceeds to the last label.
    feat_vec_ordered_by_labels, break_feat = sort_feature_using_labels(feat_vec_data, labels_data, len(config_voronoi['cls']))
    
    # Takes each column of feat_vec_ordered_by_labels and calculates its average.
    mean_feat_vec_ord = np.array(feat_vec_ordered_by_labels).mean(axis=0)
    
    # Translate each line of feat_vec_ordered_by_labels by the mean_feat_vec_ord vector.
    translated_feat_vec_ord = np.array(feat_vec_ordered_by_labels) - mean_feat_vec_ord
    
    pca = PCA(n_components = num_comp_pca)
    
    # Reduce the number of columns to build the Voronoi diagram. Remembering that the number of columns cannot be greater than the number of classes.
    pca_feat_vec_ord = pca.fit_transform(translated_feat_vec_ord)
    information_kept = pca.explained_variance_ratio_.sum()
    
    # We are splitting the pca_feat_vec_ord vector by putting all the elements of class 0 in a cell, then those of class 1, class 2, etc. 
    pca_feat_vec_ord = np.array([pca_feat_vec_ord[break_feat[i]:break_feat[i+1]] for i in range(len(break_feat)-1)])
    
    # In each class its average vector is calculated.
    feat_vec_med_resized = np.array([x.mean(axis=0) for x in pca_feat_vec_ord])
    
    assert feat_vec_med_resized.shape == (len(config_voronoi['cls']),num_comp_pca), 'the format is ({},{})'.format(len(config_voronoi['cls']),num_comp_pca)
    
    return feat_vec_med_resized, information_kept


# Function takes the most repeated color and associates the classes that are associated with that color returning a dictionary where the key is the most repeated color and the values are the associated classes.
def get_dic_vert_color(dict_vor, repeated_colors):
    vert_color_repet = {}
    for repeated_color in repeated_colors:
        vert_color_repet[repeated_color] = []
    tam = len(dict_vor)
    for i in range(tam):
        for repeated_color in repeated_colors:
            if(dict_vor[i] == repeated_color):
                vert_color_repet[repeated_color].append(config_voronoi['cls'][i]) 
    return vert_color_repet


# The function chooses the method that takes the greatest number of classes and returns information about the color graph associated with the last Voronoi diagram.
def get_informatios_graph_coloring(dataset):
    
    if(dataset == 'CIFAR10'):
        config_voronoi['cls'] = config_voronoi['CIFAR10']
        config_voronoi['path_save_data_for_voronoi'] = "/"+ os.path.join('root', 'PROJETOS', 'GCOOD', 'data', 'CIFAR10', 'data_for_const_voronoi')
    elif(dataset == 'CIFAR100'):
        config_voronoi['cls'] = config_voronoi['CIFAR100']
        config_voronoi['path_save_data_for_voronoi'] = "/"+ os.path.join('root', 'PROJETOS', 'GCOOD', 'data', 'CIFAR100', 'data_for_const_voronoi')
   
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset))
    
    # number of classes
    num_class = len(config_voronoi['cls'])
    # Taking the data to build the Voronoi diagram
    feat_vec_med_resized, _  = get_data_for_voronoi(config_voronoi['num_comp_pca'])
    
    # List of extremes of segments that cross perpendicularly the lines drawn by the Voronoi diagram.
    voronoi = Voronoi(feat_vec_med_resized)
    vor_rid = voronoi.ridge_points
    
    # List of parameters that will be used to choose the algorithm to color the graph, where each element of the list represents the choice of a different graph placement algorithm.
    list_parameters_color_graph = config_voronoi['list_parameters_color_graph']
    
    # The size of this list indicates the number of algorithms used.
    total = len(list_parameters_color_graph)
    
    # Function that builds the graph
    G = nx.Graph()
    G.add_edges_from(vor_rid)
    
    # This variable representing the number of classes to be removed
    max_class_excl = 0
    
    # This variable represents the index that was chosen from list_parameters_color_graph, 
    # which removes the largest number of classes, that is, index returns the chosen model.
    index = 0
    
    # This variable determines the number of colors used to color the graph.
    num_color_used = 0
    
    # The for will go through all the methods until it chooses the one that takes the most class number.
    for i in range(total):
        d = nx.coloring.greedy_color(G, strategy=list_parameters_color_graph[i][0], 
                             interchange=list_parameters_color_graph[i][1])
        n_cor_used = np.array(list(d.values())).max()+1
        list_count = [list(d.values()).count(j) for j in range(num_class)]
        n_class_excl = np.array(list_count).max()
        if(max_class_excl < n_class_excl):
            index = i
            max_class_excl = n_class_excl
            num_color_used = n_cor_used
            
        assert len(d) == len(config_voronoi['cls']), 'library problem nx.Graph. The color dictionary must have {} keys'.format(len(config_voronoi['cls']))
    
            
    # Dictionary that associates the vertex to its color
    dic_color = nx.coloring.greedy_color(G, strategy=list_parameters_color_graph[index][0], 
                                     interchange=list_parameters_color_graph[index][1])
    
    # Calculates which color is used most by the method.
    max_color_repeated = max([list(dic_color.values()).count(i) for i in range(num_color_used)])
    repeated_colors = [i for i in range(num_color_used) if list(dic_color.values()).count(i) == max_color_repeated]
    
    # This variable is a dictionary where the key is the most used color and the values are the classes that used that color.
    class_associated_most_used_color = get_dic_vert_color(dic_color, repeated_colors)
            
    print('The method chosen was the {}'.format(index))
    print('Number of classes to be removed by this method were equal to {}'.format(max_class_excl))
    print('Number of colors used by this method to color the graph were {} colors'.format(num_color_used))
    print('This dictionary associates each vertex with its color: {}'.format(dic_color))
    print('The most used color by this method was the color {}'.format(repeated_colors))
    print('In this {} dictionary, the key represents the most used color and the values represent the classes that used that color.'.format(class_associated_most_used_color))

          
    # Drawing the graph
    nx.draw_random(G)
    plt.show()
          
    return index, max_class_excl, num_color_used, dic_color, repeated_colors, class_associated_most_used_color
