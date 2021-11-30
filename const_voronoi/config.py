import os

config_voronoi = dict(
    num_comp_pca = 7, #Number of PCA components. Remembering that this number has to be less than the number of classes.
    path_save_data_for_voronoi = "/"+ os.path.join('root', 'PROJETOS', 'GCOOD', 'data', 'CIFAR10', 'data_for_const_voronoi'),
    list_parameters_color_graph = [("largest_first",False),("largest_first",True),("random_sequential",False),
                       ("random_sequential",True),("smallest_last",False),("smallest_last",True),
                       ("independent_set",False),("connected_sequential_bfs",False),("connected_sequential_bfs",True),
                       ("connected_sequential_dfs",False), ("connected_sequential_dfs",True), ("connected_sequential",False),
                       ("connected_sequential",True),("saturation_largest_first",False),("DSATUR",False)],
    CIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    CIFAR100 = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'],
    cls =[]
)
