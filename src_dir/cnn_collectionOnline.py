#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout
from torch.optim    import Adam, SGD



class CnnOnline(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Conv1d modules and assign them
        as member variables.
        """
        super(CnnOnline, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Assuming D_in=D_out=H
        self.Conv1   = torch.nn.Conv1d(1,int(H),D_in, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        self.Conv2   = torch.nn.Conv1d(int(H),D_out,1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        self.relu   = torch.nn.LeakyReLU().to(device)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """


        Current_batchsize=int(x.shape[0])  # N in pytorch docs
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        ConvOut1=self.relu(self.Conv1(x2.to(device)))
        ConvOut2=self.Conv2(ConvOut1) 
        y_pred = ConvOut2.view(Current_batchsize, -1) #flatten channel dimension to be 1 and get array of  dimension (batch dim,input dim)


        return y_pred

