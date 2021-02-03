#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout
from torch.optim    import Adam, SGD



class TwoLayerNet(torch.nn.Module):

    def __init__(self, D_in, H,H2, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them
        as member variables.
        """
        super(TwoLayerNet, self).__init__()

        # self.linear1 = torch.nn.Linear(D_in, H)
        # self.linear2 = torch.nn.Linear(H, D_out)

        # self.linear1 = torch.nn.Linear(D_in, H)
        # self.linear2 = torch.nn.Linear(H, H2)
        # self.linear3 = torch.nn.Linear(H2, D_out)


        # self.Conv1   = torch.nn.Conv1d(1, 10, int(D_in/10 +1), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.MaxPool1= torch.nn.MaxPool1d( int(D_in-3*int(D_in/10)+1), stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # self.linear1 = torch.nn.Linear(2*D_in, 2*D_in)
        # self.linear2 = torch.nn.Linear(2*D_in, D_out)


        
        # self.Conv1   = torch.nn.Conv1d(1,D_in,D_in, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.linear1 = torch.nn.Linear(D_in, D_out)




        # self.Conv2   = torch.nn.Conv1d(1,2,3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.MaxPool1= torch.nn.MaxPool1d(int(D_in-105), stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # self.linear1 = torch.nn.Linear(200, D_in)
        # self.linear2 = torch.nn.Linear(D_in, D_in)
        # self.linear3 = torch.nn.Linear(D_in, D_out)



        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.linear1 = torch.nn.Linear(D_in, H).to(device)
        # self.linear2 = torch.nn.Linear(H, D_out).to(device)


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Conv1   = torch.nn.Conv1d(1,int(H),D_in, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        self.Conv2   = torch.nn.Conv1d(int(H),D_in,1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        self.relu   = torch.nn.LeakyReLU().to(device)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """
        # h_relu = self.linear1(x).clamp(min=0)
        # y_pred = self.linear2(h_relu)


        # h_relu1 = self.linear1(x).clamp(min=0)
        # h_relu2 = self.linear2(h_relu1).clamp(min=0)
        # y_pred = self.linear3(h_relu1)


        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input              
        # ConvOut=self.Conv1(x2) 
        # MaxPoolOut=self.MaxPool1(ConvOut) 
        # MaxPoolOutFlat = MaxPoolOut.view(Current_batchsize, -1)
        # h_relu1 = self.linear1(MaxPoolOutFlat).clamp(min=0)
        # y_pred = self.linear2(h_relu1)


        # h_relu1 = self.linear1(x)#.clamp(min=0)
        # # h_relu2 = self.linear2(h_relu1)#.clamp(min=0)
        # y_pred = self.linear2(h_relu1)



        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut=self.Conv1(x2) 
        # ConvOutFlat = ConvOut.view(Current_batchsize, -1)
        # y_pred=self.linear1(ConvOutFlat)


        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # out1=self.linear1( x.to(device)).clamp(min=0)
        # y_pred=self.linear2(out1)



        Current_batchsize=int(x.shape[0])  # N in pytorch docs
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        ConvOut1=self.relu(self.Conv1(x2.to(device)))
        ConvOut2=self.Conv2(ConvOut1) 
        y_pred = ConvOut2.view(Current_batchsize, -1)


        return y_pred

