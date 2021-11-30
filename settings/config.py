import os

config_data_classifiers = dict(
    model = 'resnet',
    epochs=20,
    batch_size=256,
    learning_rate=0.0003,
    optimizer = 'adam',
    step_size = 190,  # at each step_size the learning_rate dacais a gamma factor
    gamma = 0.1,
    path_folder_save_weight = os.path.abspath(os.path.join('data', 'CIFAR10', 'classifiers', 'resnet', 'models', 'complete')), #saves the weights of the trained network with all labels
    path_dict_train_test = os.path.abspath(os.path.join('data', 'CIFAR10', 'data_for_input')), # pre-processed data to use in my model input
    frequency_save_model = 1,
    init_count_epoch = 1,
    network_pretrained = True,
    path_weight = None)

config_GCOOD_network = dict(
    model = 'resnet',
    epochs=150,
    batch_size=15800,
    learning_rate=0.004012,
    optimizer = 'sgd',
    activ_function = 'relu',
    dropout = 0.2757,
    feature_extraction_depth = 6,
    groups = 5,
    classification_depth = 1,
    step_size = 190,  # at each step_size the learning_rate dacais a gamma factor
    gamma = 0.1,
    path_folder_save_weight = os.path.abspath(os.path.join('data', 'CIFAR10', 'GCOOD_network', 'model')),
    path_dict_train_test = os.path.abspath(os.path.join('data', 'CIFAR10', 'data_for_input')), # pre-processed data to use in my model input
    frequency_save_model = 10,
    init_count_epoch = 1)
