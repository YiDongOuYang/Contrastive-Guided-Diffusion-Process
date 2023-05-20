import torch
import torch.optim as optim
import ipdb

import sys
sys.path.append("../adversarial_robustness_pytorch-main/")
from core.models import create_model
from core.data import get_data_info

from hard_negative_sample import criterion,EmbeddingModel
# sys.path.append("../ddim-main-contrast/")
# from adv_training.core.models import create_model
# from adv_training.core.data import get_data_info
# from adv_training.hard_negative_sample import criterion,EmbeddingModel


arg={'nb_classes':10,'model':'mnist_net_for_sample','normalize':False,
'data_dir':'/home/yidongoy/data/mnist','weight_dir':'/home/yidongoy/adversarial_robustness_pytorch-main/logAcquisition/baseline_mnist/weights-best.pt'}#/mntnfs/apmath_data1/yidong/adversarial_robustness_pytorch-main/logAcquisition/gtsrb_baseline_400/weights-best.pt

info = get_data_info(arg['data_dir'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = create_model(arg['model'], arg['normalize'], info, device)
checkpoint = torch.load(arg["weight_dir"])

model.load_state_dict(checkpoint, strict=False)#please be careful
model = torch.nn.DataParallel(model)

# model.train() #!for acquisition score
model.eval() 
del checkpoint

embedding_net = torch.nn.DataParallel(EmbeddingModel(input_size=100, output_size=64).cuda())
optimizer = optim.Adam(embedding_net.parameters(), lr=1e-2)

def get_Acquisition_grad(x,x_0):
    # ipdb.set_trace()
    with torch.enable_grad():  #It is very important
        delta = torch.zeros_like(x, requires_grad=True)
        output = criterion(embedding_net(model(x+delta)), embedding_net(model(x_0)),0.1,128,1.0, 'hard')
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