import torch

import torchvision
import torchvision.transforms as transforms

import re
import numpy as np

from .semisup import SemiSupervisedDataset
from .semisup import SemiSupervisedSampler

from .GTSRB import GTSRB


def load_gtsrbs(data_dir, use_augmentation=False, aux_take_amount=None, 
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
    data_dir = re.sub('cifar10s', 'cifar10', data_dir)
    test_transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = SemiSupervisedGTSRB10( base_dataset='gtsrb', root=data_dir, split='train', download=True, 
                                          transform=train_transform, aux_data_filename=aux_data_filename, 
                                          add_aux_labels=True, aux_take_amount=aux_take_amount, validation=validation)
    test_dataset = SemiSupervisedGTSRB10( base_dataset='gtsrb', root=data_dir, split='test', download=True, 
                                         transform=test_transform)
    if validation:
        val_dataset = SemiSupervisedGTSRB10(data_dir, split='train', download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 1024))  
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset


class SemiSupervisedGTSRB10(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for CIFAR10.
    """
    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'gtsrb', 'Only semi-supervised cifar10 is supported. Please use correct dataset!'
        self.dataset = GTSRB( **kwargs)
        self.dataset_size = len(self.dataset)
