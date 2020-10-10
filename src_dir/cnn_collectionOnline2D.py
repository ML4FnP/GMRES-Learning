#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout
from torch.optim    import Adam, SGD

import torch.nn.init as I


class CnnOnline_2D(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Conv1d modules and assign them
        as member variables.
        """
        super(CnnOnline_2D, self).__init__()
        
#         #20x20 and 25x25 dim input
        self.Conv1   = torch.nn.Conv2d(1,16,(9,9), stride=1, padding=(8,8), dilation=2, groups=1, bias=True, padding_mode='reflect')#even dim
        self.Conv2   = torch.nn.Conv2d(16,16,(7,7),stride=1, padding=(6,6), dilation=2, groups=1, bias=True, padding_mode='reflect')#even dim
        self.Conv3   = torch.nn.Conv2d(16,8,(5,5),stride=1, padding=(4,4), dilation=2, groups=1, bias=True, padding_mode='reflect')#even dim
        self.Conv4   = torch.nn.Conv2d(8,4,(3,3),stride=1, padding=(1,1), dilation=1, groups=1, bias=True, padding_mode='reflect')#even dim
        self.Conv5   = torch.nn.Conv2d(4,2,(3,3), stride=1, padding=(1,1), dilation=1, groups=1, bias=True, padding_mode='reflect')#even dim
        self.Conv6   = torch.nn.Conv2d(2,1,(1,1), stride=1, padding=(0,0), dilation=1, groups=1, bias=True, padding_mode='reflect')#even dim

        #  30x30 dim input
#         self.Conv1   = torch.nn.Conv2d(1,16,(9,9), stride=1, padding=(16,16), dilation=4, groups=1, bias=True, padding_mode='reflect')#even dim
#         self.Conv2   = torch.nn.Conv2d(16,16,(7,7),stride=1, padding=(12,12), dilation=4, groups=1, bias=True, padding_mode='reflect')#even dim
#         self.Conv3   = torch.nn.Conv2d(16,8,(5,5),stride=1, padding=(8,8), dilation=4, groups=1, bias=True, padding_mode='reflect')#even dim
#         self.Conv4   = torch.nn.Conv2d(8,4,(3,3),stride=1, padding=(3,3), dilation=3, groups=1, bias=True, padding_mode='reflect')#even dim
#         self.Conv5   = torch.nn.Conv2d(4,2,(3,3), stride=1, padding=(3,3), dilation=3, groups=1, bias=True, padding_mode='reflect')#even dim
#         self.Conv6   = torch.nn.Conv2d(2,1,(1,1), stride=1, padding=(0,0), dilation=1, groups=1, bias=True, padding_mode='reflect')#even dim

#         self.lin1 = torch.nn.Linear(int(D_in**2.0),int(D_out**2.0),bias=False)
        self.relu   = torch.nn.LeakyReLU()



    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """


#         Current_batchsize=int(x.shape[0])  # N in pytorch docs
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
#         ConvOut1=self.relu(self.Conv1(x2.to(device)))
#         ConvOut2=self.relu(self.Conv2(ConvOut1))
#         ConvOut2_squeeze = ConvOut2.squeeze(1) #Remove channel dimension
#         ConvOut2Flat=ConvOut2_squeeze.view(Current_batchsize,1,-1)
#         y_predFlat=self.lin1(ConvOut2Flat)
#         y_pred=y_predFlat.view(Current_batchsize,30,30)





#         Current_batchsize=int(x.shape[0])  # N in pytorch docs
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         xFlat=x.view(Current_batchsize,1,-1)
#         y_predFlat=self.lin1(xFlat)
#         y_pred=y_predFlat.view(Current_batchsize,30,30)


        Current_batchsize=int(x.shape[0])  # N in pytorch docs
        x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        ConvOut1=self.relu(self.Conv1(x2))
        ConvOut2=self.relu(self.Conv2(ConvOut1))
        ConvOut3=self.relu(self.Conv3(ConvOut2))
        ConvOut4=self.relu(self.Conv4(ConvOut3))
        ConvOut5=self.relu(self.Conv5(ConvOut4))
        ConvOut6=(self.Conv6(ConvOut5))
        y_pred = ConvOut6.squeeze(1) #Remove channel dimension


#         Current_batchsize=int(x.shape[0])  # N in pytorch docs
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
#         ConvOut1=self.relu(self.Conv1(x2.to(device)))
#         ConvOut2=self.relu(self.Conv2(ConvOut1))
#         ConvOut3=self.relu(self.Conv3(ConvOut2))
#         ConvOut4=self.relu(self.Conv4(ConvOut3))
#         ConvOut5=self.relu(self.Conv5(ConvOut4))
#         ConvOut6=self.relu(self.Conv6(ConvOut5))
#         ConvOut6Flat=ConvOut6.view(Current_batchsize,1,1,-1)
#         linout=self.lin1(ConvOut6Flat)
#         lin_unflat=linout.view(Current_batchsize,1,10,10)
#         y_pred=self.convT(lin_unflat)
#         y_pred = y_pred.squeeze(1) #Remove channel dimension

        


        return y_pred

    
    
#     def _initialize_weights(self):
#         I.orthogonal_(self.Conv1.weight)
#         I.orthogonal_(self.Conv2.weight)
#         I.orthogonal_(self.Conv3.weight)
#         I.orthogonal_(self.Conv4.weight)
#         I.orthogonal_(self.Conv5.weight)
#         I.orthogonal_(self.Conv6.weight)
#         I.orthogonal_(self.lin1.weight)
#         I.orthogonal_(self.convT.weight)
            