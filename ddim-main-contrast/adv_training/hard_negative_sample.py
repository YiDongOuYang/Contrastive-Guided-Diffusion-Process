import torch
import torch.nn.functional as F
from torch import nn
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
temperature = 10#0.5


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def criterion(out_1,out_2,tau_plus,batch_size,beta, estimator):
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)

        old_neg = neg.clone()
        mask = get_negative_mask(batch_size).to(device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        
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