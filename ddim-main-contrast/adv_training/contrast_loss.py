from adv_training.core.models import create_model
from adv_training.core.data import get_data_info
from adv_training.core.utils import parser_eval
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np    
import time  
from adv_training.infonce import InfoNCE
from adv_training.hard_negative_sample import criterion,EmbeddingModel
from adv_training.hard_negative_sample_conditional import criterion as criterion_conditional
from adv_training.hard_negative_sample_conditional import criterion_other_class as criterion_conditional_other_class


arg={'nb_classes':10,'model':'wrn-28-10','normalize':False,
'data_dir':'/home/yidongoy/data/cifar10','weight_dir':'/home/yidongoy/adversarial_robustness_pytorch-main/logAcquisition/deepmind/weights-best.pt'}

info = get_data_info(arg['data_dir'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = create_model(arg['model'], arg['normalize'], info, device)
checkpoint = torch.load(arg["weight_dir"])

model.load_state_dict(checkpoint, strict=False)#please be careful
model = torch.nn.DataParallel(model)

# model.train() #!for acquisition score
model.eval() 
del checkpoint

embedding_net = torch.nn.DataParallel(EmbeddingModel().cuda())
optimizer = optim.Adam(embedding_net.parameters(), lr=1e-2)

# real = np.load('/mntnfs/theory_data4/cifar_embedding.npz',allow_pickle=True)
# image_embedding = torch.from_numpy(real['embedding']).float().cuda()

# negative_pair=torch.cat((image_embedding[5000:5050],image_embedding[10000:10050],image_embedding[15000:15050],image_embedding[20000:20050],image_embedding[25000:25050],image_embedding[30000:30050],image_embedding[35000:35050],image_embedding[40000:40050],image_embedding[45000:45050]), dim=0)
# label = torch.from_numpy(real['label']).cuda()
# negative_pair_label=torch.cat((label[5000:5050],label[10000:10050],label[15000:15050],label[20000:20050],label[25000:25050],label[30000:30050],label[35000:35050],label[40000:40050],label[45000:45050]), dim=0)
# print(negative_pair_label)
''' 
def get_Acquisition_grad(x,x_0):

    # import ipdb
    # ipdb.set_trace()
    loss = InfoNCE()
    with torch.enable_grad():  #It is very important
        delta = torch.zeros_like(x, requires_grad=True)
        output = loss(model(x+delta), model(x_0))
        output.backward()
    return delta.grad*1000
'''
'''
#hard negative sample
def get_Acquisition_grad(x,x_0):

    # import ipdb
    # ipdb.set_trace()
    
    with torch.enable_grad():  #It is very important
        delta = torch.zeros_like(x, requires_grad=True)
        output = criterion(embedding_net(model(x+delta)), embedding_net(model(x_0)),0.1,512,1.0, 'hard')
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        
        a=delta.grad*200000
        #print(a)
    return a

'''
#hard negative sample + true data as positive pair for conditional generating
# def get_Acquisition_grad(x,x_0 ,real):
def get_Acquisition_grad(x,x_0):

    # import ipdb
    # ipdb.set_trace()

    with torch.enable_grad():  #It is very important
        delta = torch.zeros_like(x, requires_grad=True)
        output = criterion(embedding_net(model(x+delta)), embedding_net(model(x_0)),0.1,512,1.0, 'hard')
        #output = criterion_conditional(embedding_net(model(x+delta)), embedding_net(model(x_0)), embedding_net(real),0.1,512,1.0, 'hard')
  
        #without using embedding net
        # using x_0 as negative pair
        # output = criterion_conditional(F.normalize(model(x+delta), dim=-1), F.normalize(model(x_0), dim=-1), F.normalize(image_embedding[:512], dim=-1),0.1,512,1.0, 'hard')
        # use other class imaages as negative pair
        # output = criterion_conditional_other_class(F.normalize(model(x+delta), dim=-1), F.normalize(negative_pair, dim=-1), F.normalize(image_embedding[:512], dim=-1),0.1,512,1.0, 'hard') 
        #output = criterion_conditional(F.normalize(model(x+delta), dim=-1), F.normalize(model(x_0), dim=-1), F.normalize(real, dim=-1),0.1,500 ,1.0, 'hard') 
        
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        
        # a=delta.grad*2000000
        a=delta.grad*100000
        # print(a)
    return a
