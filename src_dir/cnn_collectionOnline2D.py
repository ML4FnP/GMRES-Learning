#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout
from torch.optim    import Adam, SGD



class CnnOnline_2D(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Conv1d modules and assign them
        as member variables.
        """
        super(CnnOnline_2D, self).__init__()
        
        # Assuming D_in=D_out=H
        self.Conv1   = torch.nn.Conv2d(1,1,D_in, stride=1, padding=int((D_in-1)/2), dilation=1, groups=1, bias=False, padding_mode='zeros') #odd dim
        # self.Conv1   = torch.nn.Conv2d(1,1,D_in-1, stride=1, padding=int((D_in-1)/2), dilation=1, groups=1, bias=False, padding_mode='zeros')#even dim
        self.lin1 = torch.nn.Linear(int(D_in**2.0),int(D_out**2.0))
        self.relu   = torch.nn.LeakyReLU()
        # self.relu   = torch.nn.PReLU()



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
        ConvOut1_squeeze = ConvOut1.squeeze(1) #Remove channel dimension
        ConvOut1Flat=ConvOut1_squeeze.view(Current_batchsize,1,-1)
        y_predFlat=self.lin1(ConvOut1Flat)
        y_pred=y_predFlat.view(Current_batchsize,25,25)

        return y_pred

