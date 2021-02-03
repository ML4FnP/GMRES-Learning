#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout
from torch.optim    import Adam, SGD

import torch.nn.init as I


class CnnOnline(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Conv1d modules and assign them
        as member variables.
        """
        super(CnnOnline, self).__init__()
        
        # Assuming D_in=D_out=H
        self.Conv1   = torch.nn.Conv1d(1,int(H),D_in, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.Conv2   = torch.nn.Conv1d(int(H),D_out,1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.relu   = torch.nn.PReLU(num_parameters=int(H))
        self.linear1 = torch.nn.Linear(D_in,D_out,bias=False)

        # self.Conv1   = torch.nn.Conv1d(1,1,int(D_in/10), stride=int(D_in/10), padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.Conv2   = torch.nn.Conv1d(1,10,10, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.Conv3   = torch.nn.Conv1d(10,32,1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.Conv4   = torch.nn.Conv1d(32,32,1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.Conv5   = torch.nn.Conv1d(32,D_in,1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.relu   = torch.nn.LeakyReLU().to(device)


        # self.linear1=torch.nn.Linear(D_in,H).to(device)
        # self.linear2=torch.nn.Linear(H,D_out).to(device)
        # self.relu   = torch.nn.LeakyReLU().to(device)


        # conv-fc-15
        # self.Conv1   = torch.nn.Conv1d(1,1,int(D_in/15), stride=int(D_in/15), padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.Conv2   = torch.nn.Conv1d(1,1,15, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.linear1 = torch.nn.Linear(15,15).to(device)
        # self.linear2 = torch.nn.Linear(15,15).to(device)
        # self.linear3 = torch.nn.Linear(15,15).to(device)
        # self.linear4 = torch.nn.Linear(15,D_out).to(device)
        # self.relu   = torch.nn.LeakyReLU().to(device)

        # self.Conv1   = torch.nn.Conv1d(1,1,int(D_in/10), stride=10, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.linear1 = torch.nn.Linear(int(D_in/10)-1,int(D_in/10)).to(device)
        # self.linear2 = torch.nn.Linear(int(D_in/10),int(D_in/10)).to(device)
        # self.linear3 = torch.nn.Linear(int(D_in/10),int(D_in/10)).to(device)
        # self.linear4 = torch.nn.Linear(int(D_in/10),D_out).to(device)
        # self.relu   = torch.nn.LeakyReLU().to(device)
        
        # self.Conv1   = torch.nn.Conv1d(1,1,int(D_in/10), stride=3, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device) # Lout= 3n/10 +1
        # self.Conv2   = torch.nn.Conv1d(1,1,5, stride=3, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device) # Lout=  floor((3n-40)/30)+1
        # self.linear1 = torch.nn.Linear(int((3*D_in-40)/30 +1),int(D_in/10)).to(device)
        # self.linear2 = torch.nn.Linear(int(D_in/10),int(D_in/10)).to(device)
        # self.linear3 = torch.nn.Linear(int(D_in/10),int(D_in/10)).to(device)
        # self.linear4 = torch.nn.Linear(int(D_in/10),D_out).to(device)
        # self.relu   = torch.nn.LeakyReLU().to(device)

        # self.Conv1   = torch.nn.Conv1d(1,1,int(D_in/4+2), stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device) 
        # self.Conv2   = torch.nn.Conv1d(1,1,int(D_in/4+1), stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device) 
        # self.Conv3   = torch.nn.Conv1d(1,1,int(D_in/4+2), stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.linear1 = torch.nn.Linear(int(D_in/4),int(D_in/2)).to(device)
        # self.linear2 = torch.nn.Linear(int(D_in/2),int(D_in/2)).to(device)
        # self.linear3 = torch.nn.Linear(int(D_in/2),int(D_in/2)).to(device)
        # self.linear4 = torch.nn.Linear(int(D_in/2),D_out).to(device)
        # self.relu   = torch.nn.LeakyReLU().to(device)

        # self.Conv1   = torch.nn.Conv1d(1,int(H),D_in, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.Conv2   = torch.nn.Conv1d(int(H),D_out,1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.relu   = torch.nn.LeakyReLU().to(device)

        # self.Conv1   = torch.nn.Conv1d(1,int(H),D_in, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.Conv2   = torch.nn.Conv1d(int(H),D_out,1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.Conv3   = torch.nn.Conv1d(int(H),D_out,1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros').to(device)
        # self.relu   = torch.nn.LeakyReLU().to(device)



    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """

        # Optimal network found
        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut1=self.relu(self.Conv1(x2.to(device)))
        # ConvOut2=self.Conv2(ConvOut1) 
        # y_pred = ConvOut2.view(Current_batchsize, -1) #flatten channel dimension to be 1 and get array of  dimension (batch dim,input dim)



        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut1=self.relu(self.Conv1(x2.to(device)))
        # ConvOut2=self.relu(self.Conv2(ConvOut1))
        # ConvOut3=self.relu(self.Conv3(ConvOut2))
        # ConvOut4=self.relu(self.Conv4(ConvOut3))
        # ConvOut5=self.Conv5(ConvOut4)
        # y_pred = ConvOut5.view(Current_batchsize, -1) #flatten channel dimension to be 1 and get array of  dimension (batch dim,input dim)


        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # LinOut1=self.relu(self.linear1(x.to(device)))
        # y_pred=self.linear2(LinOut1)

        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut1=self.relu(self.Conv1(x2.to(device)))
        # ConvOut1Flat=ConvOut1.view(Current_batchsize, -1)
        # LinOutRelu1=self.relu(self.linear1(ConvOut1Flat))
        # LinOutRelu2=self.relu(self.linear2(LinOutRelu1))
        # LinOutRelu3=self.relu(self.linear3(LinOutRelu2))
        # y_pred=self.linear4(LinOutRelu3)

        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut1=self.relu(self.Conv1(x2.to(device)))
        # ConvOut1Flat=ConvOut1.view(Current_batchsize, -1)
        # LinOutRelu1=self.relu(self.linear1(ConvOut1Flat))
        # LinOutRelu2=self.relu(self.linear2(LinOutRelu1))
        # LinOutRelu3=self.relu(self.linear3(LinOutRelu2))
        # y_pred=self.linear4(LinOutRelu3)

        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut1=self.relu(self.Conv1(x2.to(device)))
        # ConvOut2=self.relu(self.Conv2(ConvOut1))
        # ConvOut3=self.relu(self.Conv3(ConvOut2))
        # ConvOut3Flat=ConvOut3.view(Current_batchsize, -1)
        # LinOutRelu1=self.relu(self.linear1(ConvOut3Flat))
        # LinOutRelu2=self.relu(self.linear2(LinOutRelu1))
        # LinOutRelu3=self.relu(self.linear3(LinOutRelu2))
        # y_pred=self.linear4(LinOutRelu3)

        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut1=self.relu(self.Conv1(x2.to(device)))
        # ConvOut2=self.relu(self.Conv2(ConvOut1))
        # ConvOut3=self.Conv2(ConvOut2) 
        # y_pred = ConvOut3.view(Current_batchsize, -1) #flatten ch

        y_pred=self.linear1(x)

        return y_pred


    def forwardWeights(self):

        W=self.linear1.weight

        return W


    # def initialize_weights(self):
    #     I.orthogonal_(self.linear1.weight)
        
