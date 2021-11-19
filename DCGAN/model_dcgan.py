# The templates in this script were based on the pytorch website https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# The script has undergone some changes, such as image input size, learning rate and some added dropouts.
from DCGAN.config import config_gan
from DCGAN.data_utils import loady_dataset
from DCGAN.data_utils import get_dataset_for_gans
from DCGAN.data_utils import convert_for_numpy_uint8_with_label
from DCGAN.data_utils import plotting_grid_some_images
import torch
import random
from sklearn.utils import shuffle
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import time
import os

# This class is my fake image generator from my gan.
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( config_gan['nz'], config_gan['ngf'] * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config_gan['ngf'] * 16),
            nn.ReLU(True),
            # state size. (config_gan['ngf']*16) x 4 x 4
            nn.ConvTranspose2d(config_gan['ngf'] * 16, config_gan['ngf'] * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config_gan['ngf'] * 8),
            nn.ReLU(True),
            # state size. (config_gan['ngf']*8) x 8 x 8
            nn.ConvTranspose2d( config_gan['ngf'] * 8, config_gan['ngf'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config_gan['ngf'] * 4),
            nn.ReLU(True),
            # state size. (config_gan['ngf']*4) x 16 x 16
            nn.ConvTranspose2d( config_gan['ngf'] * 4, config_gan['ngf']*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config_gan['ngf']*2),
            nn.ReLU(True),
            # state size. (config_gan['ngf']) x 32 x 32
            nn.ConvTranspose2d( config_gan['ngf']*2, config_gan['nc'], 1, 1, 0, bias=False),
            nn.Tanh()
            # state size. (config_gan['nc']) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)
    
    
# This class is responsible for determining which images are fake and which are real, it's my discriminator.
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (config_gan['nc']) x 32 x 32
            nn.Conv2d(config_gan['nc'], config_gan['ndf']*2, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=config_gan['dropout']),
            # state size. (config_gan['ndf']*2) x 32 x 32
            nn.Conv2d(config_gan['ndf']*2, config_gan['ndf'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config_gan['ndf'] * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=config_gan['dropout']),
            # state size. (config_gan['ndf']*4) x 16 x 16
            nn.Conv2d(config_gan['ndf'] * 4, config_gan['ndf'] * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config_gan['ndf'] * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=config_gan['dropout']),
            # state size. (config_gan['ndf']*8) x 8 x 8
            nn.Conv2d(config_gan['ndf'] * 8, config_gan['ndf'] * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config_gan['ndf'] * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=config_gan['dropout']),
            # state size. (config_gan['ndf']*16) x 4 x 4
            nn.Conv2d(config_gan['ndf'] * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
# Custom initialization of weights, so that we have a normal distribution with mean 0.0 and standard deviation equal to 0.02.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# Gan training.        
def train_dbgan(device, dataloader, discriminator, generator, path_weight):
    
    # Choosing a random seed 
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    # Initialize BCELoss function Binary Cross Entropy loss 
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(32, config_gan['nz'], 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=config_gan['learning_rateD'], betas=(config_gan['beta1'], 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=config_gan['learning_rateG'], betas=(config_gan['beta1'], 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting the training loop...")
    # For each epoch
    for epoch in range(config_gan['init_count_epoch'], config_gan['epochs']+1):
    
        # start time epoch
        init_time = time.time()
    
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, config_gan['nz'], 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, config_gan['epochs'], i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == config_gan['epochs']) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        
        if epoch % config_gan['frequency_save_model'] == 0:
            torch.save(discriminator.state_dict(), '{}/netD_epoch{}.pt'.format(path_weight, epoch))
            torch.save(generator.state_dict(), '{}/netG_epoch{}.pt'.format(path_weight, epoch))
            
        #Time spent to run an epoch
        time_elapsed = time.time() - init_time
        # Print
        print(f'| Epoch: {epoch:02} | Time_spent: {int(time_elapsed//60):}m {int(time_elapsed % 60)}s|')
        
    return img_list, G_losses, D_losses


# This function is in charge of opening and preparing the data, training and plotting the results.
def pipeline_dcgan(device, labels_for_training, loady_cifar10_train = True, pretraining = False, save_epoch = None):
    
     # We choose whether to get Cifar training data or test data to use in training
    if(loady_cifar10_train):
        data, _ = loady_dataset()
    elif(not loady_cifar10_train):
        _, data = loady_dataset()
    
    cls = data.classes
    
    for i in labels_for_training:
        print('The dataset with the label {} of the class {} will be trained'.format(i,cls[i]))
    
    all_labels = list(data.class_to_idx.values())
    labels = [x for x in all_labels if x not in labels_for_training] # in this set we have the labels that will be excluded
    
    # Data prepared to be used in the model
    dataloader = get_dataset_for_gans(data, labels, config_gan['batch_size'])
    
    # Print some images that will be used in the template
    plotting_grid_some_images(device, dataloader, 'Some training images')
    
    # Create the generator
    netG = Generator(config_gan['ngpu'])
    
    # Create the discriminator
    netD = Discriminator(config_gan['ngpu'])
    
    # Creating a path to save model weights if the model has not yet been trained, or finding the path where weights have been saved if the model has already been trained.
    if(not pretraining):
        try:
            new_path = config_gan['path_weight'] + '/gan_{}'.format('_'.join(map(str, labels_for_training)))
            os.mkdir(new_path)
        except FileNotFoundError:
            raise FileNotFoundError('The {} directory cannot be created. Check the path entered.'.format(path_save))
        except FileExistsError:
            print('Directory {} has already been created previously.'.format(new_path))
            print('The model weights will be saved in this directory and may overwrite existing weights in it.')
            
    elif(pretraining):
        new_path = config_gan['path_weight'] + '/gan_{}'.format('_'.join(map(str, labels_for_training))) 
        path_weight_netG = '{}/netG_epoch{}.pt'.format(new_path, save_epoch)
        path_weight_netD = '{}/netD_epoch{}.pt'.format(new_path, save_epoch)
        
        # Loading with saved models
        print('Carrying the weights...')
        netG(torch.load(path_weight_netG))
        netD(torch.load(path_weight_netD))
        
    netG.to(device)
    netD.to(device)
    
    # Conditioning to multiple GPUs if desired
    if (device.type == 'cuda') and (config_gan['ngpu'] > 1):
        netG = nn.DataParallel(netG, list(range(config_gan['ngpu'])))
        
    
    # Conditioning to multiple GPUs if desired
    if (device.type == 'cuda') and (config_gan['ngpu'] > 1):
        netD = nn.DataParallel(netD, list(range(config_gan['ngpu'])))
    
    # Applies the weights_init function to randomly start all weights with mean=0 and standard deviation=0.02
    netG.apply(weights_init)
    
    # Applies the weights_init function to randomly start all weights with mean=0 and standard deviation=0.02
    netD.apply(weights_init)

    # Print the model
    print("Here we see the generator model")
    print(netG)
    
    # Print the model
    print('Here we see the discriminator model')
    print(netD)
    
    # Starting Training
    print('Starting our training')
    img_list, G_losses, D_losses = train_dbgan(device, dataloader, discriminator=netD, generator=netG, path_weight=new_path)
    
    
    # Plotting the discriminator and generator training graphs
    plotting_train_disc_gener(G_losses, D_losses)
    
    # Plotting the real and fake images
    plotting_real_images_fake_images(device, real_image= dataloader, fake_image=img_list )
    
    return netG, netD, img_list, G_losses, D_losses


# Plot of generator and discriminator error data during training.
def plotting_train_disc_gener(G_losses, D_losses):

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
# Plot of real images and fake images given by the generator.
def plotting_real_images_fake_images(device, real_image, fake_image):
    
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(real_image))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(fake_image[-1],(1,2,0)))
    plt.show()
    
# Animation of the evolution of image training.
def plotting_animation_training(img_list_train, fig_size = 12):
    
    #%%capture
    fig = plt.figure(figsize=(fig_size,fig_size))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list_train]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    return HTML(ani.to_jshtml())

# This function is for plotting real images and fakes to be used as a comparison tool.
def compare_real_fake_images(device, labels, num_epoch):
    
    # Dataset reading
    data, _ = loady_dataset()
    
    # classes
    cls = data.classes
    
    for i in labels:
        print('The dataset with the label {} of the class {} will be trained'.format(i,cls[i]))
        
    all_labels = list(data.class_to_idx.values())
    
    # labels_complementary will now be the complementary set of labels with respect to the all_labels set
    labels_complementary = [x for x in all_labels if x not in labels] 
    
    # Data prepared to be used in the model
    dataloader = get_dataset_for_gans(data, labels_complementary, config_gan['batch_size'])
    
    # Print some images that will be used in the template
    plotting_grid_some_images(device, dataloader, 'Some training images')
    
    # Creating the Generator object to generate fake images
    netG = Generator(1)
    new_path = config_gan['path_weight'] + '/gan_{}'.format('_'.join(map(str, labels))) 
    path_weight_netG = '{}/netG_epoch{}.pt'.format(new_path, num_epoch)
    netG.load_state_dict(torch.load(path_weight_netG))
    netG.to(device)
    
    # Choosing a random seed to create the latent space vector to be used by generator
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    noise = torch.randn(64, config_gan['nz'], 1, 1, device=device)
    
    # List of fake images to be plotted
    img_list= []
    
    # Creating the fake images and adding the img_list to be plotted
    with torch.no_grad():
        fake = netG(noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        
    # Plotting the images    
    plotting_real_images_fake_images(device, real_image=dataloader, fake_image=img_list)
    
    
# Getting the fake dataset with a certain label
def fake_images(device, label, num_epoch, size_batch_images):
    
    # Creating the Generator object to generate fake images
    netG = Generator(1)
    new_path = config_gan['path_weight'] + '/gan_{}'.format('_'.join(map(str, label))) 
    path_weight_netG = '{}/netG_epoch{}.pt'.format(new_path, num_epoch)
    netG.load_state_dict(torch.load(path_weight_netG))
    netG.to(device)
    
    # Choosing a random seed to create the latent space vector to be used by generator
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    noise = torch.randn(size_batch_images, config_gan['nz'], 1, 1, device=device)
    
    # Catching fake images
    with torch.no_grad():
        fake = netG(noise).detach().cpu()
        
    dataset_fake = convert_for_numpy_uint8_with_label(data = fake, label = label)
    
    return dataset_fake

# Get the fake dataset with all labels and images. However, it is necessary to configure the number of each trained epoch, as well as the number of fake images to be acquired from each class.
def get_dataset_fake_configured(device, num_epoch = 45000, size_batch_images = 10000):
    
    # Dataset reading
    data, _ = loady_dataset()
    
    # Dataset that will contain all data
    dataset = {}
    dataset['data'] = []
    dataset['targets'] = []
    
    for lab in list(data.class_to_idx.values()):
        dataset_fake = fake_images(device, label=[lab], num_epoch=num_epoch, size_batch_images=size_batch_images)
        dataset['data'].extend(dataset_fake['data'])
        dataset['targets'].extend(dataset_fake['targets'])
        dataset['data'], dataset['targets'] = shuffle(dataset['data'], dataset['targets'])
        
    return dataset

# This function just prints each result according to the label passed dataset CIFAR-10.
def get_image_results(device, label, num_epoch=45000):
    
    # dataset
    data, _ = loady_dataset()
    print("{} IMAGES".format(data.classes[label].upper()))
    compare_real_fake_images(device, labels=[label], num_epoch=num_epoch)
