'''
It is modified based on carwin's work. Original code can be found in 
https://github.com/clcarwin/focal_loss_pytorch
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int , torch.long)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        dim = input.shape[-1]
        if input.dim()>2:
            input = input.contiguous().view(-1, dim)   # N,H*W,C => N*H*W,C

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class OneHotLoss(nn.Module):
    def __init__(self):
        super(OneHotLoss, self).__init__()

    def forward(self, pred, target):
        '''
            pred is one-hot label though target does not to be like that.
        '''
        total_loss = F.nll_loss(pred, target)

        return total_loss
