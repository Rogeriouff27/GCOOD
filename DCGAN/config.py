import os

config_gan = dict(
    dataset = 'CIFAR10',
    epochs=500000,
    init_count_epoch = 1,
    batch_size=512,
    learning_rateG=0.001,
    learning_rateD=0.00002,
    dropout = 0.4,
    beta1 = 0.5, # Beta1 hyperparam for Adam optimizers
    ngpu = 1, # Number of GPUs available. Use 0 for CPU mode.
    nz = 100, # Size of z latent vector (i.e. size of generator input
    ngf = 32, # Size of feature maps in generator
    nc = 3, # Number of channels in the training images. For color images this is 3
    ndf = 32, # Size of feature maps in discriminator
    path_weight = os.path.abspath(os.path.join('data', 'CIFAR10', 'gans_model')), #./data/cifar10/gans_model/', #save models from the gan
    frequency_save_model = 100
    )