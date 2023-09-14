from adv_training.core.models import create_model
from adv_training.core.data import get_data_info
from adv_training.core.utils import parser_eval
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np    
import time  

arg={'nb_classes':10,'model':'wrn-28-10-ori','normalize':False,
'data_dir':'/home/yidongoy/data/cifar10','weight_dir':'/home/yidongoy/adversarial_robustness_pytorch-main/logAcquisition/deepmind/weights-best.pt'}

info = get_data_info(arg['data_dir'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = create_model(arg['model'], arg['normalize'], info, device)
checkpoint = torch.load(arg["weight_dir"])

model.load_state_dict(checkpoint)
model = torch.nn.DataParallel(model)
model.eval()
del checkpoint

def get_Acquisition_grad(x):

    # import ipdb
    # ipdb.set_trace()
    
    with torch.enable_grad():  #It is very important
        delta = torch.zeros_like(x, requires_grad=True)
        output = F.cross_entropy(model(x+delta), torch.zeros(x.shape[0],dtype=torch.long).cuda())

        output.backward()
        
        a=delta.grad*5000
        #print(a)
    return a