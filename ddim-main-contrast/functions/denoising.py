import torch
import adv_training.contrast_loss as contrast_loss
import adv_training.guided_by_classifier as guided_by_classifier


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


# def generalized_steps(x,real, seq, model, b, **kwargs):
def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            #if i<=73: #错误的
            #    delta= acquisition.get_Acquisition_grad(xt)
            #    et+=delta#0.1*
            
            '''
            #guided by classifier
            delta= guided_by_classifier.get_Acquisition_grad(xt)
            et+=delta
            '''

            
            #conditional generation controled by contrastive loss
            if (len(xs)>=2)&(i>=400):#&(i<600):
                # delta= contrast_loss.get_Acquisition_grad(xt,xs[-2], real)
                delta= contrast_loss.get_Acquisition_grad(xt,xs[-2])
                et+=delta
            

            '''
            #千万要注意 contrastive loss这里是加号 acquisition是减号
            if (len(xs)>=2)&(i>=400):#&(i<600):
                delta= contrast_loss.get_Acquisition_grad(xt,xs[-2])
                #print("l2 norm of et is %f"%torch.norm(et,2))
                #print("linf norm of et is %f"%torch.norm(et,float('inf')))
                #print("l2 norm of delta is %f"%torch.norm(delta,2))
                #print("linf norm of delta is %f"%torch.norm(delta,float('inf')))
                et+=delta
            '''
              
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

   
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et   #I need also check the gradient norm
            xs.append(xt_next.to('cpu'))
            '''
            print("\n norm eta")
            print(torch.norm(et,float('inf')))
            print("\n origin norm is")
            print(torch.norm(xt_next,float('inf')))
            '''
          
    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
