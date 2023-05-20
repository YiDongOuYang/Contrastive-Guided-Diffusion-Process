import torch

import torchvision
import torchvision.transforms as transforms

import re
import numpy as np

from .semisup_our import SemiSupervisedDataset
from .semisup_our import SemiSupervisedSampler

from .StanfordCars_semi import StanfordCars


def load_stanfordcarss(data_dir, use_augmentation=False, aux_take_amount=None, 
                  aux_data_filename='/home/luoqijun/yidong/data/cifar10_ddpm.npz', 
                  validation=False):
    """
    Returns semisupervised CIFAR10 train, test datasets and dataloaders (with Tiny Images).
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
        aux_take_amount (int): number of semi-supervised examples to use (if None, use all).
        aux_data_filename (str): path to additional data pickle file.
    Returns:
        train dataset, test dataset. 
    """
    data_dir = re.sub('stanfordcarss', 'stanfordcars', data_dir)   
    test_transform = transforms.Compose([ transforms.Resize([224, 224]),transforms.ToTensor(),transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.Resize(250),transforms.transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                              transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))])
    else: 
        train_transform = test_transform
    
    train_dataset = SemiSupervisedStanfordCars(base_dataset='stanfordcars',root=data_dir,  train=True, download=True, 
                                          transform=train_transform, aux_data_filename=aux_data_filename, 
                                          add_aux_labels=True, aux_take_amount=aux_take_amount, validation=validation)
    test_dataset = SemiSupervisedStanfordCars(base_dataset='stanfordcars',root=data_dir,  train=False, download=True, 
                                         transform=test_transform)
    if validation:
        val_dataset = StanfordCars(root=data_dir, train=True, download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 1024))  
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset


class SemiSupervisedStanfordCars(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for CIFAR10.
    """
    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'stanfordcars', 'Only semi-supervised cifar10 is supported. Please use correct dataset!'
        self.dataset = StanfordCars(**kwargs)
        self.dataset_size = len(self.dataset)
