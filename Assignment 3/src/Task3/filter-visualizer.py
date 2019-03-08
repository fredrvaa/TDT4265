import dataloaders
import numpy as np
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms, datasets, utils
from torch import nn
import torchvision
import os
import utils
import matplotlib.pyplot as plt

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

#Dataloader for normal image
def load_cifar10_standard(batch_size, validation_fraction=0.1):
    transform = [
        transforms.ToTensor(),
    ]

    transform = transforms.Compose(transform)
    data_train = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform)

    train_indices = list(range(len(data_train)))

    train_sampler = SequentialSampler(train_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2)

    return dataloader_train

#Dataloader for normalized image (could have just normalized loaded image)
def load_cifar10_normalized(batch_size, validation_fraction=0.1):
    transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]

    transform = transforms.Compose(transform)
    data_train_normalized = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform)

    train_indices = list(range(len(data_train_normalized)))

    train_sampler = SequentialSampler(train_indices)

    dataloader_train_normalized = torch.utils.data.DataLoader(data_train_normalized,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2)

    return dataloader_train_normalized

#Activation images
def first_layer(model, image):
    image = nn.functional.interpolate(image, size=(256, 256))
    image = model.conv1(image)
    return image

def last_layer(model, image):
    image = nn.functional.interpolate(image, size=(256, 256))
    im = image
    for i, child in enumerate(model.children()):
        im = child(im)
        if (i > 7): 
          return im

#Visualizer function
def visualize_filters(model):
    #Load normal set
    train = load_cifar10_standard(6)
    image = next(iter(train))[0][0]
    image = image.numpy().transpose(1,2,0)
    #Normalize image
    image_normalized = torchvision.transforms.functional.normalize(image.data, mean, std)

    #Load normalized set
    train_normalized = load_cifar10_normalized(6)


    os.makedirs("filters", exist_ok=True)

    #Visualize first layer filters
    for i in range(0,10):
        for (X_batch, Y_batch) in (train_normalized):

            image_features = first_layer(model, utils.to_cuda(X_batch)).cpu() 

            im = image_features[0,i,:,:].detach()
            plt.imsave('filters/first_layer_im' + str(i) + '.eps', im)
            plt.imshow(im)
            plt.show()

            break

    #Visualize last layer filters
    for i in range(0,10):
        for (X_batch, Y_batch) in (train_normalized):

            image_features = last_layer(model, utils.to_cuda(X_batch)).cpu()

            im = image_features[0,i,:,:].detach()
            plt.imsave('filters/last_layer_im' + str(i) + '.eps', im)
            plt.imshow(im)
            plt.show()

            break