import os

config_voronoi = dict(
    path_save_data_for_voronoi = os.path.abspath(os.path.join('data', 'cifar10', 'data_for_const_voronoi')),
    list_parameters_color_graph = [("largest_first",False),("largest_first",True),("random_sequential",False),
                       ("random_sequential",True),("smallest_last",False),("smallest_last",True),
                       ("independent_set",False),("connected_sequential_bfs",False),("connected_sequential_bfs",True),
                       ("connected_sequential_dfs",False), ("connected_sequential_dfs",True), ("connected_sequential",False),
                       ("connected_sequential",True),("saturation_largest_first",False),("DSATUR",False)],
    cls = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
)
