import torch

import torchvision
import torchvision.transforms as transforms
import numpy as np


DATA_DESC = {
    'data': 'stanfordcars',
    "num_classes": 196,
    # 'mean': [0.3403, 0.3121, 0.3214], 
    # 'std': [0.2724, 0.2608, 0.2669],
}


def load_stanfordcars(data_dir, use_augmentation=False,validation=False):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([ transforms.Scale(224),transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.Scale(250),transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(),transforms.ToTensor() ])
    else: 
        train_transform = test_transform
    
    train_dataset = torchvision.datasets.StanfordCars(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.StanfordCars(root=data_dir, train=False, download=True, transform=test_transform)    
    if validation:
        val_dataset = torchvision.datasets.StanfordCars(root=data_dir, train=True, download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 1024))  
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset