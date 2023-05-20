import torch

import torchvision
import torchvision.transforms as transforms
import numpy as np


DATA_DESC = {
    'data': 'mnist',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
}


def load_mnist(data_dir, use_augmentation=False,validation=False):
    """
    Returns MNIST train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(
                28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)

    #training on a subset
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, 5000)) 
    
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)    
    if validation:
        val_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 1024))  
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset