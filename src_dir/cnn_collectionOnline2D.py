#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout
from torch.optim    import Adam, SGD

from math import floor, log2


#TODO: is the import needed?
import torch.nn.init as I



class FluidNet2D10(torch.nn.Module):


    def __init__(self, D_in, D_out):
        """
        FluidNet 2D optimized fro a 10-19 dim resolution.
        """

        super().__init__()
        #TODO: make compatible with multiple devices
        #TODO: this should be class member
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pad_7Kernel = torch.nn.ZeroPad2d(3)
        self.pad_5Kernel = torch.nn.ZeroPad2d(2)
        self.pad_3Kernel = torch.nn.ZeroPad2d(1)

        self.AVG = torch.nn.AvgPool2d(2, stride=2)
        self.Upsample = torch.nn.Upsample(mode='bilinear', size=(D_in, D_in))

        self.ConvInit1 = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=7, bias=False)
        self.ConvInit2 = torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, bias=False)

        self.Conv11 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, bias=False)
        self.Conv12 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, bias=False)
        self.Conv21 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, bias=False)
        self.Conv22 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, bias=False)
        self.Conv31 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, bias=False)
        self.Conv32 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, bias=False)

        self.Conv_Post1 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, bias=False)
        self.Conv_Post2 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, bias=False)
        self.Conv_Post3 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, bias=False)
        self.Conv_Post4 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, bias=False)

        self.Conv4 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, bias=False)
        self.Conv5 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, bias=False)

        self.ConvOut = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1,bias=False)

        self.relu   = torch.nn.LeakyReLU()


    def forward(self, x,DataSetSize,Factor):
        x2=x.unsqueeze(1)  # Add channel dimension (C) to input
        Current_batchsize=int(x.shape[0])  # N in pytorch docs

        if (DataSetSize < Factor*1):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_3Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_3Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(z3))
            z3 = self.relu(self.Conv32(z3))
            z3 = self.Upsample(z3)
            # Sum all convolution output scales
            z = z1 + z2 + z3
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)

        # Forward function with 1 extra resblock at end
        if (DataSetSize >= Factor*1):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_3Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_3Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(z3))
            z3 = self.relu(self.Conv32(z3))
            z3 = self.Upsample(z3)
            # Sum all convolution output scales
            z = z1 + z2 + z3
            #resblock 1
            y = self.relu(z)
            y = self.relu(self.Conv_Post1(self.pad_3Kernel(z)))
            y = self.Conv_Post2(self.pad_3Kernel(y))
            z = z + y
            z = self.relu(z)
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)

        # Forward function with 2 extra resblock at end
        if (DataSetSize >= Factor*2):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_3Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_3Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(z3))
            z3 = self.relu(self.Conv32(z3))
            z3 = self.Upsample(z3)
            # Sum all convolution output scales
            z = z1 + z2 + z3
            #resblock 1
            y = self.relu(z)
            y = self.relu(self.Conv_Post1(self.pad_3Kernel(z)))
            y = self.Conv_Post2(self.pad_3Kernel(y))
            z = z + y
            z = self.relu(z)
            #resblock 2
            y = self.relu(z)
            y = self.relu(self.Conv_Post3(self.pad_3Kernel(z)))
            y = self.Conv_Post4(self.pad_3Kernel(y))
            z = z + y
            z = self.relu(z)
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)



class FluidNet2D20(torch.nn.Module):


    def __init__(self, D_in, D_out):
        """
        FluidNet 2D optimized for a 20-29 dim resolution.
        """

        super().__init__()
        #TODO: make compatible with multiple devices
        #TODO: this should be class member
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pad_7Kernel = torch.nn.ZeroPad2d(6)
        self.pad_5Kernel = torch.nn.ZeroPad2d(4)
        self.pad_3Kernel = torch.nn.ZeroPad2d(1)
        self.pad_3Kernel_Dilated = torch.nn.ZeroPad2d(2)

        self.AVG = torch.nn.AvgPool2d(2, stride=2)
        self.Upsample = torch.nn.Upsample(mode='bilinear', size=(D_in, D_in))

        self.ConvInit1 = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=7, dilation=2, bias=False)
        self.ConvInit2 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, dilation=2, bias=False)

        self.Conv11 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, dilation=2, bias=False)
        self.Conv12 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, dilation=2, bias=False)
        self.Conv21 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, dilation=2, bias=False)
        self.Conv22 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, dilation=2, bias=False)
        self.Conv31 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, bias=False)
        self.Conv32 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, bias=False)
        self.Conv41 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, bias=False)
        self.Conv42 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, bias=False)

        self.Conv_Post1 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=2, bias=False)
        self.Conv_Post2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=2, bias=False)
        self.Conv_Post3 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=2, bias=False)
        self.Conv_Post4 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=2, bias=False)
        self.Conv_Post5 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=2, bias=False)
        self.Conv_Post6 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=2, bias=False)
        self.Conv_Post7 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=2, bias=False)
        self.Conv_Post8 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=2, bias=False)


        self.Conv4 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, bias=False)
        self.Conv5 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, bias=False)

        self.ConvOut = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False)

        self.relu = torch.nn.LeakyReLU()

        self.FDpad = torch.nn.ZeroPad2d(1)

        self.Aweights = torch.tensor(
            [[0.25, 0.5, 0.25],
             [0.5, -3., 0.5],
             [0.25,  0.5, 0.25]]
        )
        self.Aweights = self.Aweights.view(1, 1, 3 ,3)
        self.Aweights = self.Aweights.to(device)

        self.Blur = (1/16)*torch.tensor(
            [[1., 2., 1.],
             [2., 4., 2.],
             [1.,  2., 1.]]
        )
        self.Blur = self.Blur.view(1, 1, 3 ,3)
        self.Blur = self.Blur.to(device)


    def forward(self, x, DataSetSize, Factor):
        x2 = x.unsqueeze(1)  # Add channel dimension (C) to input
        Current_batchsize = int(x.shape[0])  # N in pytorch docs


        x3=torch.nn.functional.conv2d(self.FDpad(x2), self.Blur, bias=None, stride=1)
        x4=torch.nn.functional.conv2d(self.FDpad(x2), self.Aweights, bias=None, stride=1)
        x2=torch.cat((x2,x3),1)
        x2=torch.cat((x2,x4),1)

        if (DataSetSize < Factor*1):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            z4 = self.AVG(z3)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(self.pad_3Kernel(z3)))
            z3 = self.relu(self.Conv32(self.pad_3Kernel(z3)))
            z3 = self.Upsample(z3)
            # Downsample3 scale
            z4 = self.relu(self.Conv41((z4)))
            z4 = self.relu(self.Conv42((z4)))
            z4 = self.Upsample(z4)
            # Sum all convolution output scales
            z = z1 + z2 + z3 + z4
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)

        # Forward function with 1 extra resblock at end
        if (DataSetSize >= Factor*1):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            z4 = self.AVG(z3)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(self.pad_3Kernel(z3)))
            z3 = self.relu(self.Conv32(self.pad_3Kernel(z3)))
            z3 = self.Upsample(z3)
            # Downsample3 scale
            z4 = self.relu(self.Conv41((z4)))
            z4 = self.relu(self.Conv42((z4)))
            z4 = self.Upsample(z4)
            # Sum all convolution output scales
            z = z1 + z2 + z3 + z4
            #resblock 1
            y = self.relu(z)
            y = self.relu(self.Conv_Post1(self.pad_3Kernel_Dilated(z)))
            y = self.Conv_Post2(self.pad_3Kernel_Dilated(y))
            z = z + y
            z = self.relu(z)
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)

        # Forward function with 2 extra resblock at end
        if (DataSetSize >= Factor*2):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            z4 = self.AVG(z3)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(self.pad_3Kernel(z3)))
            z3 = self.relu(self.Conv32(self.pad_3Kernel(z3)))
            z3 = self.Upsample(z3)
            # Downsample3 scale
            z4 = self.relu(self.Conv41((z4)))
            z4 = self.relu(self.Conv42((z4)))
            z4 = self.Upsample(z4)
            # Sum all convolution output scales
            z = z1 + z2 + z3 + z4
            #resblock 1
            y = self.relu(z)
            y = self.relu(self.Conv_Post1(self.pad_3Kernel_Dilated(z)))
            y = self.Conv_Post2(self.pad_3Kernel_Dilated(y))
            z = z + y
            z = self.relu(z)
            #resblock 2
            y = self.relu(z)
            y = self.relu(self.Conv_Post3(self.pad_3Kernel_Dilated(z)))
            y = self.Conv_Post4(self.pad_3Kernel_Dilated(y))
            z = z + y
            z = self.relu(z)
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)

        # Forward function with 3 extra resblock at end
        if (DataSetSize >= Factor*3):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            z4 = self.AVG(z3)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(self.pad_3Kernel(z3)))
            z3 = self.relu(self.Conv32(self.pad_3Kernel(z3)))
            z3 = self.Upsample(z3)
            # Downsample3 scale
            z4 = self.relu(self.Conv41((z4)))
            z4 = self.relu(self.Conv42((z4)))
            z4 = self.Upsample(z4)
            # Sum all convolution output scales
            z = z1+z2+z3+z4
            #resblock 1
            y = self.relu(z)
            y = self.relu(self.Conv_Post1(self.pad_3Kernel_Dilated(z)))
            y = self.Conv_Post2(self.pad_3Kernel_Dilated(y))
            z = z + y
            z = self.relu(z)
            #resblock 2
            y = self.relu(z)
            y = self.relu(self.Conv_Post3(self.pad_3Kernel_Dilated(z)))
            y = self.Conv_Post4(self.pad_3Kernel_Dilated(y))
            z = z + y
            #resblock 3
            y = self.relu(z)
            y = self.relu(self.Conv_Post5(self.pad_3Kernel_Dilated(z)))
            y = self.Conv_Post6(self.pad_3Kernel_Dilated(y))
            z = z + y
            z = self.relu(z)
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)

        # Forward function with 4 extra resblock at end
        if (DataSetSize >= Factor*4):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            z4 = self.AVG(z3)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(self.pad_3Kernel(z3)))
            z3 = self.relu(self.Conv32(self.pad_3Kernel(z3)))
            z3 = self.Upsample(z3)
            # Downsample3 scale
            z4 = self.relu(self.Conv41((z4)))
            z4 = self.relu(self.Conv42((z4)))
            z4 = self.Upsample(z4)
            # Sum all convolution output scales
            z = z1 + z2 + z3 + z4
            #resblock 1
            y = self.relu(z)
            y = self.relu(self.Conv_Post1(self.pad_3Kernel_Dilated(z)))
            y = self.Conv_Post2(self.pad_3Kernel_Dilated(y))
            z = z + y
            z = self.relu(z)
            #resblock 2
            y = self.relu(z)
            y = self.relu(self.Conv_Post3(self.pad_3Kernel_Dilated(z)))
            y = self.Conv_Post4(self.pad_3Kernel_Dilated(y))
            z = z + y
            #resblock 3
            y = self.relu(z)
            y = self.relu(self.Conv_Post5(self.pad_3Kernel_Dilated(z)))
            y = self.Conv_Post6(self.pad_3Kernel_Dilated(y))
            z = z + y
            #resblock 4
            y = self.relu(z)
            y = self.relu(self.Conv_Post7(self.pad_3Kernel_Dilated(z)))
            y = self.Conv_Post8(self.pad_3Kernel_Dilated(y))
            z = z + y
            z = self.relu(z)
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)



class FluidNet2D30(torch.nn.Module):


    def __init__(self, D_in, D_out):
        """
        FluidNet 2D optimized for 30-39 dim resolution.
        """

        super().__init__()
        #TODO: make compatible with multiple devices
        #TODO: this should be class member
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pad_7Kernel = torch.nn.ZeroPad2d(9)
        self.pad_5Kernel = torch.nn.ZeroPad2d(6)
        self.pad_3Kernel_3scale = torch.nn.ZeroPad2d(2)
        self.pad_3Kernel_4scale = torch.nn.ZeroPad2d(1)
        self.pad_3Kernel_Res = torch.nn.ZeroPad2d(3)

        self.AVG = torch.nn.AvgPool2d(2, stride=2)
        self.Upsample = torch.nn.Upsample(mode='bilinear', size=(D_in, D_in))

        self.ConvInit1 = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=7, dilation=3, bias=False)
        self.ConvInit2 = torch.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, dilation=3, bias=False)

        self.Conv11 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, dilation=3, bias=False)
        self.Conv12 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, dilation=3, bias=False)
        self.Conv21 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, dilation=3, bias=False)
        self.Conv22 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, dilation=3, bias=False)
        self.Conv31 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=2, bias=False)
        self.Conv32 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=2, bias=False)
        self.Conv41 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, bias=False)
        self.Conv42 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, bias=False)

        self.Conv_Post1 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=3, bias=False)
        self.Conv_Post2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=3, bias=False)
        self.Conv_Post3 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=3, bias=False)
        self.Conv_Post4 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=3, bias=False)
        self.Conv_Post5 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=3, bias=False)
        self.Conv_Post6 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=3, bias=False)
        self.Conv_Post7 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=3, bias=False)
        self.Conv_Post8 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=3, bias=False)

        self.Conv4 =torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, bias=False)
        self.Conv5 =torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, bias=False)

        self.ConvOut =torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False)

        self.relu = torch.nn.LeakyReLU()


    def forward(self, x,DataSetSize,Factor):
        x2=x.unsqueeze(1)  # Add channel dimension (C) to input
        Current_batchsize=int(x.shape[0])  # N in pytorch docs

        if (DataSetSize < Factor*1):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            z4 = self.AVG(z3)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(self.pad_3Kernel_3scale(z3)))
            z3 = self.relu(self.Conv32(self.pad_3Kernel_3scale(z3)))
            z3 = self.Upsample(z3)
            # Downsample3 scale
            z4 = self.relu(self.Conv41((self.pad_3Kernel_4scale(z4))))
            z4 = self.relu(self.Conv42((self.pad_3Kernel_4scale(z4))))
            z4 = self.Upsample(z4)
            # Sum all convolution output scales
            z = z1 + z2 + z3 + z4
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)

        # Forward function with 1 extra resblock at end
        if (DataSetSize >= Factor*1):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            z4 = self.AVG(z3)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(self.pad_3Kernel_3scale(z3)))
            z3 = self.relu(self.Conv32(self.pad_3Kernel_3scale(z3)))
            z3 = self.Upsample(z3)
            # Downsample3 scale
            z4 = self.relu(self.Conv41((self.pad_3Kernel_4scale(z4))))
            z4 = self.relu(self.Conv42(( self.pad_3Kernel_4scale(z4))))
            z4 = self.Upsample(z4)
            # Sum all convolution output scales
            z = z1 + z2 + z3 + z4
            #resblock 1
            y = self.relu(z)
            y = self.relu(self.Conv_Post1(self.pad_3Kernel_Res(z)))
            y = self.Conv_Post2(self.pad_3Kernel_Res(y))
            z = z + y
            z = self.relu(z)
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)

        # Forward function with 2 extra resblock at end
        if (DataSetSize >= Factor*2):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            z4 = self.AVG(z3)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(self.pad_3Kernel_3scale(z3)))
            z3 = self.relu(self.Conv32(self.pad_3Kernel_3scale(z3)))
            z3 = self.Upsample(z3)
            # Downsample3 scale
            z4 = self.relu(self.Conv41(( self.pad_3Kernel_4scale(z4))))
            z4 = self.relu(self.Conv42((z4)))
            z4 = self.Upsample(z4)
            # Sum all convolution output scales
            z = z1 + z2 + z3 + z4
            #resblock 1
            y = self.relu(z)
            y = self.relu(self.Conv_Post1(self.pad_3Kernel_Res(z)))
            y = self.Conv_Post2(self.pad_3Kernel_Res(y))
            z = z + y
            z = self.relu(z)
            #resblock 2
            y = self.relu(z)
            y = self.relu(self.Conv_Post3(self.pad_3Kernel_Res(z)))
            y = self.Conv_Post4(self.pad_3Kernel_Res(y))
            z = z + y
            z = self.relu(z)
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)

        # Forward function with 3 extra resblock at end
        if (DataSetSize >= Factor*3):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            z4 = self.AVG(z3)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(self.pad_3Kernel_3scale(z3)))
            z3 = self.relu(self.Conv32(self.pad_3Kernel_3scale(z3)))
            z3 = self.Upsample(z3)
            # Downsample3 scale
            z4 = self.relu(self.Conv41(( self.pad_3Kernel_4scale(z4))))
            z4 = self.relu(self.Conv42(( self.pad_3Kernel_4scale(z4))))
            z4 = self.Upsample(z4)
            # Sum all convolution output scales
            z = z1 + z2 + z3 + z4
            #resblock 1
            y = self.relu(z)
            y = self.relu(self.Conv_Post1(self.pad_3Kernel_Res(z)))
            y = self.Conv_Post2(self.pad_3Kernel_Res(y))
            z = z + y
            z = self.relu(z)
            #resblock 2
            y = self.relu(z)
            y = self.relu(self.Conv_Post3(self.pad_3Kernel_Res(z)))
            y = self.Conv_Post4(self.pad_3Kernel_Res(y))
            z = z + y
            #resblock 3
            y = self.relu(z)
            y = self.relu(self.Conv_Post5(self.pad_3Kernel_Res(z)))
            y = self.Conv_Post6(self.pad_3Kernel_Res(y))
            z = z + y
            z = self.relu(z)
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)

        # Forward function with 4 extra resblock at end
        if (DataSetSize >= Factor*4):
            z = self.ConvInit1(self.pad_7Kernel(x2))
            z1 = self.relu(self.ConvInit2(self.pad_5Kernel(z)))
            z2 = self.AVG(z1)
            z3 = self.AVG(z2)
            z4 = self.AVG(z3)
            # Full scale
            z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
            z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
            # Downsample1 scale
            z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
            z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
            z2 = self.Upsample(z2)
            # Downsample2 scale
            z3 = self.relu(self.Conv31(self.pad_3Kernel_3scale(z3)))
            z3 = self.relu(self.Conv32(self.pad_3Kernel_3scale(z3)))
            z3 = self.Upsample(z3)
            # Downsample3 scale
            z4 = self.relu(self.Conv41(( self.pad_3Kernel_4scale(z4))))
            z4 = self.relu(self.Conv42(( self.pad_3Kernel_4scale(z4))))
            z4 = self.Upsample(z4)
            # Sum all convolution output scales
            z = z1 + z2 + z3 + z4
            #resblock 1
            y = self.relu(z)
            y = self.relu(self.Conv_Post1(self.pad_3Kernel_Res(z)))
            y = self.Conv_Post2(self.pad_3Kernel_Res(y))
            z = z + y
            z = self.relu(z)
            #resblock 2
            y = self.relu(z)
            y = self.relu(self.Conv_Post3(self.pad_3Kernel_Res(z)))
            y = self.Conv_Post4(self.pad_3Kernel_Res(y))
            z = z + y
            #resblock 3
            y = self.relu(z)
            y = self.relu(self.Conv_Post5(self.pad_3Kernel_Res(z)))
            y = self.Conv_Post6(self.pad_3Kernel_Res(y))
            z = z + y
            #resblock 4
            y = self.relu(z)
            y = self.relu(self.Conv_Post7(self.pad_3Kernel_Res(z)))
            y = self.Conv_Post8(self.pad_3Kernel_Res(y))
            z = z + y
            z = self.relu(z)
            z = self.relu(self.Conv4(z))
            z = self.relu(self.Conv5(z))
            z = self.ConvOut(z)
            return z.squeeze(1)


def const_shape_pad(conv):
    pad = tuple(
        int((k + (k-1)*(d-1) - 1)/2)
        for k, d in zip(conv.kernel_size, conv.dilation)
    )
    return torch.nn.ZeroPad2d((pad[0], pad[0], pad[1], pad[1]))


class Conv2dSame(torch.nn.Conv2d):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad = const_shape_pad(self)
    
    def __call__(self, *args, **kwargs):
        return super().__call__(self.pad(*args, **kwargs))


class FluidNet2DN(torch.nn.Module):


    def __init__(self, D_in, D_out, N):
        """
        FluidNet 2D optimized for N-dim resolution.
        """

        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.AVG = torch.nn.AvgPool2d(2, stride=2)
        self.Upsample = torch.nn.Upsample(mode='bilinear', size=(D_in, D_in))

        self.ConvInit1 = Conv2dSame(
            in_channels=1, out_channels=2, kernel_size=7, dilation=3, bias=False
        )
        self.ConvInit2 = Conv2dSame(
            in_channels=2, out_channels=8, kernel_size=5, dilation=3, bias=False
        )

        self.n_conv_layers = floor(log2(N))
        self.conv = list()
        for i in range(self.n_conv_layers):
            dilation    = self.n_conv_layers - i
            kernel_size = min(3, 2 + dilation)
            self.conv.append(
                Conv2dSame(
                    in_channels=8, out_channels=8,
                    kernel_size=kernel_size, dilation=dilation, bias=False
                )
            )

        self.conv_post = Conv2dSame(
            in_channels=8, out_channels=8,
            kernel_size=min(3, 2 + self.n_conv_layers),
            dilation=self.n_conv_layers,
            bias=False
        )

        self.Conv4 = Conv2dSame(
            in_channels=8, out_channels=8, kernel_size=1, bias=False
        )
        self.Conv5 = Conv2dSame(
            in_channels=8, out_channels=1, kernel_size=1, bias=False
        )

        self.ConvOut = Conv2dSame(
            in_channels=1, out_channels=1, kernel_size=1, bias=False
        )

        self.relu = torch.nn.LeakyReLU()


    def forward(self, x, DataSetSize, Factor):
        x2 = x.unsqueeze(1)  # Add channel dimension (C) to input
        # Current_batchsize = int(x.shape[0])  # N in pytorch docs

        zn = list()

        z = self.ConvInit1(x2)
        z1 = self.relu(self.ConvInit2(z))
        zn.append(z1)
        zp = z1

        for i in range(self.n_conv_layers):
            zp = self.AVG(zp)
            zn.append(zp)

        for i in range(self.n_conv_layers):
            zp = zn[i]
            zp = self.relu(self.conv[i](zp))
            zp = self.Upsample(zp)
            zn[i] = zp

        z = zn[0]
        for i in range(1, self.n_conv_layers):
            z = z + zn[i]

        n_resnet = floor(DataSetSize / Factor)
        for i in range(n_resnet):
            y = self.relu(z)
            y = self.relu(self.conv_post(z))
            y = self.conv_post(y)
            z = z + y
            z = self.relu(z)

        z = self.relu(self.Conv4(z))
        z = self.relu(self.Conv5(z))
        z = self.ConvOut(z)

        return z.squeeze(1)


class CNN_30(torch.nn.Module):


    def __init__(self, D_in, D_out):
        """
        Sequential convolutions 2D optimized for 30-40 dim resolution.
        """

        super().__init__()
        #TODO: make compatible with multiple devices
        #TODO: this should be class member
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Conv1   = torch.nn.Conv2d(1,16,(9,9), stride=1, padding=(16,16), dilation=4, groups=1, bias=True, padding_mode='reflect')#even dim
        self.Conv2   = torch.nn.Conv2d(16,16,(7,7),stride=1, padding=(12,12), dilation=4, groups=1, bias=True, padding_mode='reflect')#even dim
        self.Conv3   = torch.nn.Conv2d(16,8,(5,5),stride=1, padding=(8,8), dilation=4, groups=1, bias=True, padding_mode='reflect')#even dim
        self.Conv4   = torch.nn.Conv2d(8,4,(3,3),stride=1, padding=(3,3), dilation=3, groups=1, bias=True, padding_mode='reflect')#even dim
        self.Conv5   = torch.nn.Conv2d(4,2,(3,3), stride=1, padding=(3,3), dilation=3, groups=1, bias=True, padding_mode='reflect')#even dim
        self.Conv6   = torch.nn.Conv2d(2,1,(1,1), stride=1, padding=(0,0), dilation=1, groups=1, bias=True, padding_mode='reflect')#even dim
        self.relu = torch.nn.LeakyReLU()



    def forward(self, x, DataSetSize, Factor):

        Current_batchsize=int(x.shape[0])  # N in pytorch docs
        x2=x.unsqueeze(1)  # Add channel dimension (C) to input
        ConvOut1=self.relu(self.Conv1(x2))
        ConvOut2=self.relu(self.Conv2(ConvOut1))
        ConvOut3=self.relu(self.Conv3(ConvOut2))
        ConvOut4=self.relu(self.Conv4(ConvOut3))
        ConvOut5=self.relu(self.Conv5(ConvOut4))
        ConvOut6=(self.Conv6(ConvOut5))
        y_pred = ConvOut6.squeeze(1) #Remove channel dimension

        return y_pred.squeeze(1)


class SingleDenseLayer(torch.nn.Module):


    def __init__(self, D_in, D_out):
        """
        Sequential convolutions 2D optimized for 30-40 dim resolution.
        """

        super().__init__()
        #TODO: make compatible with multiple devices
        #TODO: this should be class member
        device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Lin1   = torch.nn.Linear(D_in*D_in, D_out*D_out, bias=False)



    def forward(self, x, DataSetSize, Factor):

        Current_batchsize=int(x.shape[0])  # N in pytorch docs
        TensorDim = x.shape[1]
        x=torch.reshape(x,(Current_batchsize,1,-1))
        x = self.Lin1(x)
        x=torch.reshape(x,(Current_batchsize,TensorDim,TensorDim))

        return x

