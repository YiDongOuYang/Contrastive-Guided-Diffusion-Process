import torch
import torch.nn.functional as F
from torch import nn
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
temperature = 10#0.5


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size,  batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
    
    return negative_mask

#out_2 as x_0, which is the last diffusion step 
def criterion(out_1,out_2,true,tau_plus,batch_size,beta, estimator):
        # neg score
        
        neg = torch.exp(torch.mm(out_1, out_2.t().contiguous()) / temperature)

        mask = get_negative_mask(batch_size).to(device)
        neg = neg.masked_select(mask).view(batch_size, -1)

        # pos score
        pos = torch.exp(torch.mm(out_1, true.t().contiguous()) / temperature)
        
        # negative samples similarity scoring
        if estimator=='hard':
            N = batch_size  - 1
            neg = torch.clamp(neg, min=1e-4)
            imp = (beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        elif estimator=='easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
        
        ratio = torch.clamp(pos/(pos + Ng), min=1e-4)
        #print(torch.where(torch.isnan(ratio)))
        # contrastive loss
        loss = - (torch.log(ratio)).mean()
        if torch.isnan(loss).sum()>0:
            print(torch.isnan(ratio).sum())
            print(torch.isnan(Ng).sum())
            print(torch.isnan(pos).sum())
            print(Ng)
            print(pos)
            print(imp.shape)
            print(torch.isnan(imp).sum())
            exit()
        # print(loss)
        # print(ratio)
        # print(pos)
        # print(Ng)
        return loss
    
#out_2 as the embedding of other class
def criterion_other_class(out_1,out_2,true,tau_plus,batch_size,beta, estimator):
        # neg score
        
        neg = torch.exp(torch.mm(out_1, out_2.t().contiguous()) / temperature)

        # pos score
        pos = torch.exp(torch.mm(out_1, true.t().contiguous()) / temperature)
        
        # negative samples similarity scoring
        if estimator=='hard':
            N = batch_size * 2 - 2
            neg = torch.clamp(neg, min=1e-4)
            imp = (beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        elif estimator=='easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
        
        ratio = torch.clamp(pos/(pos + Ng), min=1e-4)
        #print(torch.where(torch.isnan(ratio)))
        # contrastive loss
        loss = - (torch.log(ratio)).mean()
        if torch.isnan(loss).sum()>0:
            print(torch.isnan(ratio).sum())
            print(torch.isnan(Ng).sum())
            print(torch.isnan(pos).sum())
            print(Ng)
            print(pos)
            print(imp.shape)
            print(torch.isnan(imp).sum())
            exit()
        # print(loss)
        # print(ratio)
        # print(pos)
        # print(Ng)
        return loss

class EmbeddingModel(nn.Module):
    def __init__(self, input_size=640, output_size=128):
        super(EmbeddingModel, self).__init__() 
        self.encoder = nn.Linear(input_size, output_size)
        self.decoder = nn.Linear(output_size, output_size)
        
    def forward(self, input):
        output = F.relu(self.encoder(input))
        output = self.decoder(output)
        return F.normalize(output, dim=-1)