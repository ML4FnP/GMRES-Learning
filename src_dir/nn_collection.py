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

        # self.linear1 = torch.nn.Linear(D_in, H)
        # self.linear2 = torch.nn.Linear(H, H2)
        # self.linear3 = torch.nn.Linear(H2, H2)
        # self.linear4 = torch.nn.Linear(H2, H2)
        # self.linear5 = torch.nn.Linear(H2, H2)
        # self.linear6 = torch.nn.Linear(H2, D_out)


        # self.Conv1   = torch.nn.Conv1d(1, 10, int(D_in/10 +1), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.MaxPool1= torch.nn.MaxPool1d( int(D_in-3*int(D_in/10)+1), stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # self.linear1 = torch.nn.Linear(2*D_in, 2*D_in)
        # self.linear2 = torch.nn.Linear(2*D_in, D_out)


        # self.Conv1   = torch.nn.Conv1d(1,1,5, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.Conv2   = torch.nn.Conv1d(1,2,3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.MaxPool1= torch.nn.MaxPool1d(int(D_in-105), stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # self.linear1 = torch.nn.Linear(200, D_in)
        # self.linear2 = torch.nn.Linear(D_in, D_in)
        # self.linear3 = torch.nn.Linear(D_in, D_out)


        
        self.Conv1   = torch.nn.Conv1d(1,D_in,D_in, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')


        # self.Conv2   = torch.nn.Conv1d(1,2,3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.MaxPool1= torch.nn.MaxPool1d(int(D_in-105), stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # self.linear1 = torch.nn.Linear(200, D_in)
        # self.linear2 = torch.nn.Linear(D_in, D_in)
        # self.linear3 = torch.nn.Linear(D_in, D_out)


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
        # h_relu3 = self.linear3(h_relu2).clamp(min=0)
        # h_relu4 = self.linear4(h_relu3).clamp(min=0)
        # h_relu5 = self.linear5(h_relu4).clamp(min=0)
        # y_pred = self.linear6(h_relu5)



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



        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut1=self.Conv1(x2) 
        # ConvOut2=self.Conv2(ConvOut1) 
        # MaxPoolOut=self.MaxPool1(ConvOut2) 
        # ConvOut2Flat = MaxPoolOut.view(Current_batchsize, -1)
        # h_relu1 = self.linear1(ConvOut2Flat).clamp(min=0)
        # h_relu2 = self.linear2(h_relu1).clamp(min=0)
        # y_pred = self.linear3(h_relu2)

        # h_relu1 = self.linear1(x)#.clamp(min=0)
        # # h_relu2 = self.linear2(h_relu1)#.clamp(min=0)
        # y_pred = self.linear2(h_relu1)



        Current_batchsize=int(x.shape[0])  # N in pytorch docs
        x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        ConvOut=self.Conv1(x2) 
        y_pred = ConvOut.view(Current_batchsize, -1)

        return y_pred

        # print('unsqueeze shape',x2.shape)
        # print('ConvOut shape',ConvOut.shape)
        # print('MaxPoolOut shape',MaxPoolOut.shape)
        # print('MaxPoolOutFlat shape',MaxPoolOutFlat.shape)
        # print('h_relu1 shape',h_relu1.shape)
        # print('y_pred shape',y_pred.shape)
