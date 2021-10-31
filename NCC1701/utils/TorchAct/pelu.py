import torch as th 
import torch.nn as nn 

class PELU3(nn.Module):
    #implimented according to the paper arXiv:1605.09332v4 [cs.LG] 10 Jan 2018
    def __init__(self, a=1., b=1., c=1.):
        super().__init__()
        self.a = nn.Parameter(th.tensor(a), requires_grad=True)
        self.b = nn.Parameter(th.tensor(b), requires_grad=True)
        self.c = nn.Parameter(th.tensor(b), requires_grad=True)

    def forward(self, inputs):
        a = th.abs(self.a)
        b = th.abs(self.b) 
        c = th.abs(self.b) 
        res = th.where(inputs >= 0, c * inputs, a * (th.exp(inputs / b) - 1))
        return res 

class PELU2(nn.Module):
    #implumented following online code
    def __init__(self, a=1., b=1.):
        super().__init__()
        self.a = nn.Parameter(th.tensor(a), requires_grad=True)
        self.b = nn.Parameter(th.tensor(b), requires_grad=True)

    def forward(self, inputs):
        a = th.abs(self.a)
        b = th.abs(self.b) 
        res = th.where(inputs >= 0, a/b * inputs, a * (th.exp(inputs / b) - 1))
        return res 


class PELU1(nn.Module):
    #converting the original ELU to PELU
    def __init__(self, a=1.):
        super().__init__()
        self.a = nn.Parameter(th.tensor(a), requires_grad=True)

    def forward(self, inputs):
        a = th.abs(self.a)
        res = th.where(inputs >= 0, a * inputs, a * (th.exp(inputs) - 1))
        return res 