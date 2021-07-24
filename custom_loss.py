import torch
from torch import nn
import torch.nn.functional as F



class SigmoidLoss(nn.Module):
    
    def forward(self, p_scores, n_scores):
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()
        
        return (p_loss + n_loss) / 2, p_loss, n_loss 
