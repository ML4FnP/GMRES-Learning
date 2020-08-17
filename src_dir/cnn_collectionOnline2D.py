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
        # self.Conv1   = torch.nn.Conv2d(1,1,D_in-1, stride=1, padding=int((D_in-1)/2), dilation=1, groups=1, bias=False, padding_mode='zeros')#even dim
        # self.Conv2   = torch.nn.Conv2d(1,1,D_in-1, stride=1, padding=int((D_in-1)/2), dilation=1, groups=1, bias=False, padding_mode='zeros')#even dim
        # self.lin1 = torch.nn.Linear(int(D_in**2.0),int(D_out**2.0))
        # self.relu   = torch.nn.LeakyReLU()

        self.Conv1   = torch.nn.Conv2d(1,16,(7,7), stride=1, padding=(6,6), dilation=2, groups=1, bias=False, padding_mode='zeros')#even dim
        self.Conv2   = torch.nn.Conv2d(16,8,(5,5),stride=1, padding=(4,4), dilation=2, groups=1, bias=False, padding_mode='zeros')#even dim
        self.Conv3   = torch.nn.Conv2d(8,8,(5,5),stride=1, padding=(4,4), dilation=2, groups=1, bias=False, padding_mode='zeros')#even dim
        self.Conv4   = torch.nn.Conv2d(8,4,(3,3),stride=1, padding=(2,2), dilation=2, groups=1, bias=False, padding_mode='zeros')#even dim
        self.Conv5   = torch.nn.Conv2d(4,2,(3,3), stride=1, padding=(2,2), dilation=2, groups=1, bias=False, padding_mode='zeros')#even dim
        self.Conv6   = torch.nn.Conv2d(2,1,(1,1), stride=1, padding=(0,0), dilation=1, groups=1, bias=False, padding_mode='zeros')#even dim
        
        
        self.lin1 = torch.nn.Linear(int(D_in**2.0),int(D_out**2.0),bias=False)
        self.relu   = torch.nn.LeakyReLU()



    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """



        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut1=self.relu(self.Conv1(x2.to(device)))
        # ConvOut1_squeeze = ConvOut1.squeeze(1) #Remove channel dimension
        # ConvOut1Flat=ConvOut1_squeeze.view(Current_batchsize,1,-1)
        # y_predFlat=self.lin1(ConvOut1Flat)
        # y_pred=y_predFlat.view(Current_batchsize,20,20)


        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut1=self.relu(self.Conv1(x2.to(device)))
        # ConvOut2=self.relu(self.Conv2(ConvOut1))
        # ConvOut3=self.relu(self.Conv3(ConvOut2))
        # ConvOut3_squeeze = ConvOut3.squeeze(1) #Remove channel dimension
        # ConvOut3Flat=ConvOut3_squeeze.view(Current_batchsize,1,-1)
        # y_predFlat=self.lin1(ConvOut3Flat)
        # y_pred=y_predFlat.view(Current_batchsize,20,20)

        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut1=self.relu(self.Conv1(x2.to(device)))
        # ConvOut2=self.relu(self.Conv2(ConvOut1))
        # ConvOut3=self.relu(self.Conv3(ConvOut2))
        # ConvOut4=self.relu(self.Conv4(ConvOut3))
        # ConvOut4_squeeze = ConvOut4.squeeze(1) #Remove channel dimension
        # ConvOut4Flat=ConvOut4_squeeze.view(Current_batchsize,1,-1)
        # y_predFlat=self.lin1(ConvOut4Flat)
        # y_pred=y_predFlat.view(Current_batchsize,20,20)

        # Current_batchsize=int(x.shape[0])  # N in pytorch docs
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
        # ConvOut1=self.relu(self.Conv1(x2.to(device)))
        # ConvOut2=self.relu(self.Conv2(ConvOut1))
        # ConvOut2_squeeze = ConvOut2.squeeze(1) #Remove channel dimension
        # ConvOut2Flat=ConvOut2_squeeze.view(Current_batchsize,1,-1)
        # linout1=self.relu(self.lin1(ConvOut2Flat))
        # y_predFlat=self.relu(self.lin2(linout1))
        # y_pred=y_predFlat.view(Current_batchsize,20,20)






#######################################################



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


        return y_pred

