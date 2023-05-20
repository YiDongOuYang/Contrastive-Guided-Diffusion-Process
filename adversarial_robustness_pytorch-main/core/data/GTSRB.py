import torch

import torchvision
import torchvision.transforms as transforms
import numpy as np

from .GTSRB import GTSRB


DATA_DESC = {
    'data': 'gtsrb',
    "num_classes": 43,
    'mean': [0.3403, 0.3121, 0.3214], 
    'std': [0.2724, 0.2608, 0.2669],

    "image_size": 32,
    "train_images": 39252,
    "val_images": 12631,
    "num_channels": 3,
}


def load_gtsrb(data_dir, use_augmentation=False,validation=False):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = GTSRB(root=data_dir, split='train', download=True, transform=train_transform)
    test_dataset = GTSRB(root=data_dir, split='test', download=True, transform=test_transform)    
    if validation:
        val_dataset = GTSRB(root=data_dir, split='train', download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 1024))  
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset