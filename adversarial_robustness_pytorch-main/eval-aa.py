"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack
    
from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed

import ipdb

# Setup

parse = parser_eval()
args = parse.parse_args()

LOG_DIR = args.log_dir + args.desc
# LOG_DIR = "/home/yidongoy/adversarial_robustness_pytorch-main/logAcquisition/infoNCE_true_2M_lanet_entropy_random_1M_1024"
with open(LOG_DIR+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

DATA_DIR = args.data_dir + args.data + '/'
#LOG_DIR = args.log_dir + args.desc

WEIGHTS = LOG_DIR + '/weights-best.pt'

log_path = LOG_DIR + '/log-aa.log'
logger = Logger(log_path)
 
info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))



# Load data

seed(args.seed)
_, _,_, train_dataloader, test_dataloader,_ = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False,validation=True)
# _, _, train_dataloader, test_dataloader= load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    # shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)



# Model

model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
# ipdb.set_trace()
if 'tau' in args and args.tau:
    print ('Using WA model.')
#model.load_state_dict(checkpoint['model_state_dict'])

model = torch.nn.DataParallel(model)
'''
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    name = k[:7]+k[9:]
    new_state_dict[name] = v    
model.load_state_dict(new_state_dict)
'''
#print(checkpoint['model_state_dict'])

#for cifar10
# model.load_state_dict(checkpoint['model_state_dict'])
# for gtsrb
model.load_state_dict(checkpoint)

model.eval() 
#model.train()
del checkpoint



# AA Evaluation

seed(args.seed)
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
adversary = AutoAttack(model, norm=norm, eps=args.attack_eps, log_path=log_path, version=args.version, seed=args.seed)

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-t' ] #'apgd-ce'
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

with torch.no_grad():
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)

print ('Script Completed.')
