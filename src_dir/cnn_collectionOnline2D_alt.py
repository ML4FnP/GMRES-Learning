#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout
from torch.optim    import Adam, SGD
import torch.nn.init as I


"""
    Implementing a small reblock based CNN with a 'pine tree' configuration. 
    This is experimental! 
"""

class CnnOnline_2D(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Conv1d modules and assign them
        as member variables.
        """
        super(CnnOnline_2D, self).__init__()
        
        # Assuming D_in=D_out=H

#       Reflection Padding method, pad3 is for conv2D with kernel_size 7
#       pad2 is kernel_size 5, pad1 is kernel_size 3 

        self.pad3   = torch.nn.ReflectionPad2d(3)
        self.pad2   = torch.nn.ReflectionPad2d(2)
        self.pad1   = torch.nn.ReflectionPad2d(1)

#       Pine tree style network 

        self.Conv1  = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7)
        self.Conv2  = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5)
        self.Conv3  = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5)
        self.Conv4  = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3)
        self.Conv5  = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3)
        self.Conv6  = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        self.relu   = torch.nn.LeakyReLU()



    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """
        Current_batchsize=int(x.shape[0])  # N in pytorch docs
        x2=x.unsqueeze(1)  # Add channel dimension (C) to input
        #input layer
        z = self.Conv1(self.pad3(x2))

#       ResBlocks
#       block 1
        y = self.relu(self.Conv2(self.pad2(z)))
        y = self.Conv3(self.pad2(y))
        z = z + y
#       block 2
        y = self.relu(self.Conv4(self.pad1(z)))
        y = self.Conv5(self.pad1(y))
        z = z + y

#       Consolidating convolution and output
        z = self.Conv6(z)
        return z.squeeze(1) 

    def _initialize_weights(self):
        I.orthogonal_(self.Conv1.weight)
        I.orthogonal_(self.Conv2.weight)
        I.orthogonal_(self.Conv3.weight)
        I.orthogonal_(self.Conv4.weight)
        I.orthogonal_(self.Conv5.weight)
        I.orthogonal_(self.Conv6.weight)
