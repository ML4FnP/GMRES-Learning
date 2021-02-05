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

    def __init__(self, D_in, D_out):
        """
        In the constructor we instantiate two nn.Conv1d modules and assign them
        as member variables.
        """
        super(CnnOnline_2D, self).__init__()

########################################################################3
## FLUID NET

        # # 10-19 dim resolution
        # self.pad_7Kernel   = torch.nn.ZeroPad2d(3)
        # self.pad_5Kernel   = torch.nn.ZeroPad2d(2)
        # self.pad_3Kernel   = torch.nn.ZeroPad2d(1)

        # self.AVG = torch.nn.AvgPool2d(2, stride=2)
        # self.Upsample = torch.nn.Upsample(mode='bilinear',size=(D_in,D_in))

        # self.ConvInit1  = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=7,bias=False)
        # self.ConvInit2  = torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5,bias=False)

        # self.Conv11  = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5,bias=False)
        # self.Conv12  = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5,bias=False)
        # self.Conv21  = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,bias=False)
        # self.Conv22  = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,bias=False)
        # self.Conv31  = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1,bias=False)
        # self.Conv32  = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1,bias=False)

        # self.Conv_Post1 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,bias=False)
        # self.Conv_Post2 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,bias=False)
        # self.Conv_Post3 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,bias=False)
        # self.Conv_Post4 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,bias=False)

        # self.Conv4 =torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1,bias=False)
        # self.Conv5 =torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1,bias=False)

        # self.ConvOut =torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1,bias=False)


        # self.relu   = torch.nn.LeakyReLU()


##____________________________________________________________________
        # # 20-29 dim resolution
        self.pad_7Kernel   = torch.nn.ZeroPad2d(6)
        self.pad_5Kernel   = torch.nn.ZeroPad2d(4)
        self.pad_3Kernel   = torch.nn.ZeroPad2d(1)
        self.pad_3Kernel_Dilated   = torch.nn.ZeroPad2d(2)

        self.AVG = torch.nn.AvgPool2d(2, stride=2)
        self.Upsample = torch.nn.Upsample(mode='bilinear',size=(D_in,D_in))


        self.ConvInit1  = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=7,dilation=2,bias=False)
        self.ConvInit2  = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5,dilation=2,bias=False)

        self.Conv11  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,dilation=2,bias=False)
        self.Conv12  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,dilation=2,bias=False)
        self.Conv21  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,dilation=2,bias=False)
        self.Conv22  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,dilation=2,bias=False)
        self.Conv31  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False)
        self.Conv32  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False)
        self.Conv41  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1,bias=False)
        self.Conv42  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1,bias=False)

        self.Conv_Post1 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=2,bias=False)
        self.Conv_Post2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=2,bias=False)
        self.Conv_Post3 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=2,bias=False)
        self.Conv_Post4 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=2,bias=False)
        self.Conv_Post5 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=2,bias=False)
        self.Conv_Post6 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=2,bias=False)
        self.Conv_Post7 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=2,bias=False)
        self.Conv_Post8 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=2,bias=False)


        self.Conv4 =torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1,bias=False)
        self.Conv5 =torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1,bias=False)

        self.ConvOut =torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1,bias=False)


        self.relu   = torch.nn.LeakyReLU()


        self.FDpad=torch.nn.ZeroPad2d(1)

        self.Aweights = torch.tensor([[0.25, 0.5, 0.25],
                        [0.5, -3., 0.5],
                        [0.25,  0.5, 0.25]])
        self.Aweights = self.Aweights.view(1,1,3 ,3)
        self.Aweights= self.Aweights.to("cuda:0")
        

  
        self.Blur = (1/16)*torch.tensor([[1., 2., 1.],
                        [2., 4., 2.],
                        [1.,  2., 1.]])
        self.Blur = self.Blur.view(1,1,3 ,3)
        self.Blur= self.Blur.to("cuda:0")
##____________________________________________________________________
        # 30-39 dim resolution
        # self.pad_7Kernel   = torch.nn.ZeroPad2d(9)
        # self.pad_5Kernel   = torch.nn.ZeroPad2d(6)
        # self.pad_3Kernel_3scale   = torch.nn.ZeroPad2d(2)
        # self.pad_3Kernel_4scale   = torch.nn.ZeroPad2d(1)
        # self.pad_3Kernel_Res   = torch.nn.ZeroPad2d(3)



        # self.AVG = torch.nn.AvgPool2d(2, stride=2)
        # self.Upsample = torch.nn.Upsample(mode='bilinear',size=(D_in,D_in))


        # self.ConvInit1  = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=7,dilation=3,bias=False)
        # self.ConvInit2  = torch.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5,dilation=3,bias=False)

        # self.Conv11  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,dilation=3,bias=False)
        # self.Conv12  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,dilation=3,bias=False)
        # self.Conv21  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,dilation=3,bias=False)
        # self.Conv22  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,dilation=3,bias=False)
        # self.Conv31  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=2,bias=False)
        # self.Conv32  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=2,bias=False)
        # self.Conv41  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False)
        # self.Conv42  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False)

        # self.Conv_Post1 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=3,bias=False)
        # self.Conv_Post2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=3,bias=False)
        # self.Conv_Post3 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=3,bias=False)
        # self.Conv_Post4 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=3,bias=False)
        # self.Conv_Post5 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=3,bias=False)
        # self.Conv_Post6 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=3,bias=False)
        # self.Conv_Post7 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=3,bias=False)
        # self.Conv_Post8 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,dilation=3,bias=False)


        # self.Conv4 =torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1,bias=False)
        # self.Conv5 =torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1,bias=False)

        # self.ConvOut =torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1,bias=False)


        # self.relu   = torch.nn.LeakyReLU()

########################################################################
########################################################################

#__________________________________________________________________
## Fluidnet + resnet forward
#__________________________________________________________________


##_______________________________________________________________
# 10 dim flownet
#     def forward(self, x,DataSetSize,Factor):
#         x2=x.unsqueeze(1)  # Add channel dimension (C) to input
#         Current_batchsize=int(x.shape[0])  # N in pytorch docs


#         if (DataSetSize < Factor*1):
#                 z = self.ConvInit1(self.pad_7Kernel(x2))
#                 z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
#                 z2 =  self.AVG(z1)
#                 z3 =  self.AVG(z2)
#                 # Full scale
#                 z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
#                 z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
#                 # Downsample1 scale
#                 z2 = self.relu(self.Conv21(self.pad_3Kernel(z2)))
#                 z2 = self.relu(self.Conv22(self.pad_3Kernel(z2)))
#                 z2 = self.Upsample(z2)
#                 # Downsample2 scale
#                 z3 = self.relu(self.Conv31(z3))
#                 z3 = self.relu(self.Conv32(z3))
#                 z3 = self.Upsample(z3)
#                 # Sum all convolution output scales
#                 z = z1+z2+z3
#                 z = self.relu(self.Conv4(z))
#                 z = self.relu(self.Conv5(z))
#                 z=self.ConvOut(z)
#                 return z.squeeze(1) 


#         # Forward function with 1 extra resblock at end
#         if (DataSetSize >=Factor*1):
#                 z = self.ConvInit1(self.pad_7Kernel(x2))
#                 z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
#                 z2 =  self.AVG(z1)
#                 z3 =  self.AVG(z2)
#                 # Full scale
#                 z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
#                 z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
#                 # Downsample1 scale
#                 z2 = self.relu(self.Conv21(self.pad_3Kernel(z2)))
#                 z2 = self.relu(self.Conv22(self.pad_3Kernel(z2)))
#                 z2 =  self.Upsample(z2)
#                 # Downsample2 scale
#                 z3 = self.relu(self.Conv31(z3))
#                 z3 = self.relu(self.Conv32(z3))
#                 z3 = self.Upsample(z3)
#                 # Sum all convolution output scales
#                 z = z1+z2+z3
#                 #resblock 1
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post1(self.pad_3Kernel(z)))
#                 y = self.Conv_Post2(self.pad_3Kernel(y))
#                 z = z + y   
#                 z = self.relu(z)
#                 z = self.relu(self.Conv4(z))
#                 z = self.relu(self.Conv5(z))
#                 z=self.ConvOut(z)
#                 return z.squeeze(1) 



#         # Forward function with 2 extra resblock at end
#         if (DataSetSize >=Factor*2):
#                 z = self.ConvInit1(self.pad_7Kernel(x2))
#                 z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
#                 z2 =  self.AVG(z1)
#                 z3 =  self.AVG(z2)
#                 # Full scale
#                 z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
#                 z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
#                 # Downsample1 scale
#                 z2 = self.relu(self.Conv21(self.pad_3Kernel(z2)))
#                 z2 = self.relu(self.Conv22(self.pad_3Kernel(z2)))
#                 z2 =  self.Upsample(z2)
#                 # Downsample2 scale
#                 z3 = self.relu(self.Conv31(z3))
#                 z3 = self.relu(self.Conv32(z3))
#                 z3 = self.Upsample(z3)
#                 # Sum all convolution output scales
#                 z = z1+z2+z3
#                 #resblock 1
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post1(self.pad_3Kernel(z)))
#                 y = self.Conv_Post2(self.pad_3Kernel(y))
#                 z = z + y   
#                 z = self.relu(z)
#                 #resblock 2
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post3(self.pad_3Kernel(z)))
#                 y = self.Conv_Post4(self.pad_3Kernel(y))
#                 z = z + y    
#                 z = self.relu(z)
#                 z = self.relu(self.Conv4(z))
#                 z = self.relu(self.Conv5(z))
#                 z=self.ConvOut(z)
#                 return z.squeeze(1) 

#_____________________________________________________-
## Fluidnet forward 20 dim
    def forward(self, x,DataSetSize,Factor):
        x2=x.unsqueeze(1)  # Add channel dimension (C) to input
        Current_batchsize=int(x.shape[0])  # N in pytorch docs


        x3=torch.nn.functional.conv2d(self.FDpad(x2), self.Blur, bias=None, stride=1)
        x4=torch.nn.functional.conv2d(self.FDpad(x2), self.Aweights, bias=None, stride=1)
        x2=torch.cat((x2,x3),1)
        x2=torch.cat((x2,x4),1)

        if (DataSetSize < Factor*1):
                z = self.ConvInit1(self.pad_7Kernel(x2))
                z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
                z2 =  self.AVG(z1)
                z3 =  self.AVG(z2)
                z4 =  self.AVG(z3)
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
                z = self.relu(self.Conv4(z))
                z = self.relu(self.Conv5(z))
                z=self.ConvOut(z)
                return z.squeeze(1) 


        # Forward function with 1 extra resblock at end
        if (DataSetSize >=Factor*1):
                z = self.ConvInit1(self.pad_7Kernel(x2))
                z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
                z2 =  self.AVG(z1)
                z3 =  self.AVG(z2)
                z4 =  self.AVG(z3)
                # Full scale
                z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
                z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
                # Downsample1 scale
                z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
                z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
                z2 =  self.Upsample(z2)
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
                z = self.relu(self.Conv4(z))
                z = self.relu(self.Conv5(z))
                z=self.ConvOut(z)
                return z.squeeze(1) 



        # Forward function with 2 extra resblock at end
        if (DataSetSize >=Factor*2):
                z = self.ConvInit1(self.pad_7Kernel(x2))
                z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
                z2 =  self.AVG(z1)
                z3 =  self.AVG(z2)
                z4 =  self.AVG(z3)
                # Full scale
                z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
                z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
                # Downsample1 scale
                z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
                z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
                z2 =  self.Upsample(z2)
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
                z = self.relu(z)
                z = self.relu(self.Conv4(z))
                z = self.relu(self.Conv5(z))
                z=self.ConvOut(z)
                return z.squeeze(1) 

        # Forward function with 2 extra resblock at end
        if (DataSetSize >=Factor*3):
                z = self.ConvInit1(self.pad_7Kernel(x2))
                z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
                z2 =  self.AVG(z1)
                z3 =  self.AVG(z2)
                z4 =  self.AVG(z3)
                # Full scale
                z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
                z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
                # Downsample1 scale
                z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
                z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
                z2 =  self.Upsample(z2)
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
                z=self.ConvOut(z)
                return z.squeeze(1) 

        # Forward function with 2 extra resblock at end
        if (DataSetSize >=Factor*4):
                z = self.ConvInit1(self.pad_7Kernel(x2))
                z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
                z2 =  self.AVG(z1)
                z3 =  self.AVG(z2)
                z4 =  self.AVG(z3)
                # Full scale
                z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
                z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
                # Downsample1 scale
                z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
                z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
                z2 =  self.Upsample(z2)
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
                #resblock 4
                y = self.relu(z)
                y = self.relu(self.Conv_Post7(self.pad_3Kernel_Dilated(z)))
                y = self.Conv_Post8(self.pad_3Kernel_Dilated(y))
                z = z + y      
                z = self.relu(z)
                z = self.relu(self.Conv4(z))
                z = self.relu(self.Conv5(z))
                z=self.ConvOut(z)
                return z.squeeze(1) 


#_____________________________________________________-
## Fluidnet forward 30-39 dim
#     def forward(self, x,DataSetSize,Factor):
#         x2=x.unsqueeze(1)  # Add channel dimension (C) to input
#         Current_batchsize=int(x.shape[0])  # N in pytorch docs


#         if (DataSetSize < Factor*1):
#                 z = self.ConvInit1(self.pad_7Kernel(x2))
#                 z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
#                 z2 =  self.AVG(z1)
#                 z3 =  self.AVG(z2)
#                 z4 =  self.AVG(z3)
#                 # Full scale
#                 z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
#                 z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
#                 # Downsample1 scale
#                 z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
#                 z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
#                 z2 = self.Upsample(z2)
#                 # Downsample2 scale
#                 z3 = self.relu(self.Conv31(self.pad_3Kernel_3scale(z3)))
#                 z3 = self.relu(self.Conv32(self.pad_3Kernel_3scale(z3)))
#                 z3 = self.Upsample(z3)
#                 # Downsample3 scale
#                 z4 = self.relu(self.Conv41((self.pad_3Kernel_4scale(z4))))
#                 z4 = self.relu(self.Conv42((self.pad_3Kernel_4scale(z4))))
#                 z4 = self.Upsample(z4)
#                 # Sum all convolution output scales
#                 z = z1+z2+z3+z4
#                 z = self.relu(self.Conv4(z))
#                 z = self.relu(self.Conv5(z))
#                 z=self.ConvOut(z)
#                 return z.squeeze(1) 


#         # Forward function with 1 extra resblock at end
#         if (DataSetSize >=Factor*1):
#                 z = self.ConvInit1(self.pad_7Kernel(x2))
#                 z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
#                 z2 =  self.AVG(z1)
#                 z3 =  self.AVG(z2)
#                 z4 =  self.AVG(z3)
#                 # Full scale
#                 z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
#                 z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
#                 # Downsample1 scale
#                 z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
#                 z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
#                 z2 =  self.Upsample(z2)
#                 # Downsample2 scale
#                 z3 = self.relu(self.Conv31(self.pad_3Kernel_3scale(z3)))
#                 z3 = self.relu(self.Conv32(self.pad_3Kernel_3scale(z3)))
#                 z3 = self.Upsample(z3)
#                 # Downsample3 scale
#                 z4 = self.relu(self.Conv41((self.pad_3Kernel_4scale(z4))))
#                 z4 = self.relu(self.Conv42(( self.pad_3Kernel_4scale(z4))))
#                 z4 = self.Upsample(z4)
#                 # Sum all convolution output scales
#                 z = z1+z2+z3+z4
#                 #resblock 1
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post1(self.pad_3Kernel_Res(z)))
#                 y = self.Conv_Post2(self.pad_3Kernel_Res(y))
#                 z = z + y   
#                 z = self.relu(z)
#                 z = self.relu(self.Conv4(z))
#                 z = self.relu(self.Conv5(z))
#                 z=self.ConvOut(z)
#                 return z.squeeze(1) 



#         # Forward function with 2 extra resblock at end
#         if (DataSetSize >=Factor*2):
#                 z = self.ConvInit1(self.pad_7Kernel(x2))
#                 z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
#                 z2 =  self.AVG(z1)
#                 z3 =  self.AVG(z2)
#                 z4 =  self.AVG(z3)
#                 # Full scale
#                 z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
#                 z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
#                 # Downsample1 scale
#                 z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
#                 z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
#                 z2 =  self.Upsample(z2)
#                 # Downsample2 scale
#                 z3 = self.relu(self.Conv31(self.pad_3Kernel_3scale(z3)))
#                 z3 = self.relu(self.Conv32(self.pad_3Kernel_3scale(z3)))
#                 z3 = self.Upsample(z3)
#                 # Downsample3 scale
#                 z4 = self.relu(self.Conv41(( self.pad_3Kernel_4scale(z4))))
#                 z4 = self.relu(self.Conv42((z4)))
#                 z4 = self.Upsample(z4)
#                 # Sum all convolution output scales
#                 z = z1+z2+z3+z4
#                 #resblock 1
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post1(self.pad_3Kernel_Res(z)))
#                 y = self.Conv_Post2(self.pad_3Kernel_Res(y))
#                 z = z + y   
#                 z = self.relu(z)
#                 #resblock 2
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post3(self.pad_3Kernel_Res(z)))
#                 y = self.Conv_Post4(self.pad_3Kernel_Res(y))
#                 z = z + y    
#                 z = self.relu(z)
#                 z = self.relu(self.Conv4(z))
#                 z = self.relu(self.Conv5(z))
#                 z=self.ConvOut(z)
#                 return z.squeeze(1) 

#         # Forward function with 2 extra resblock at end
#         if (DataSetSize >=Factor*3):
#                 z = self.ConvInit1(self.pad_7Kernel(x2))
#                 z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
#                 z2 =  self.AVG(z1)
#                 z3 =  self.AVG(z2)
#                 z4 =  self.AVG(z3)
#                 # Full scale
#                 z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
#                 z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
#                 # Downsample1 scale
#                 z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
#                 z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
#                 z2 =  self.Upsample(z2)
#                 # Downsample2 scale
#                 z3 = self.relu(self.Conv31(self.pad_3Kernel_3scale(z3)))
#                 z3 = self.relu(self.Conv32(self.pad_3Kernel_3scale(z3)))
#                 z3 = self.Upsample(z3)
#                 # Downsample3 scale
#                 z4 = self.relu(self.Conv41(( self.pad_3Kernel_4scale(z4))))
#                 z4 = self.relu(self.Conv42(( self.pad_3Kernel_4scale(z4))))
#                 z4 = self.Upsample(z4)
#                 # Sum all convolution output scales
#                 z = z1+z2+z3+z4
#                 #resblock 1
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post1(self.pad_3Kernel_Res(z)))
#                 y = self.Conv_Post2(self.pad_3Kernel_Res(y))
#                 z = z + y   
#                 z = self.relu(z)
#                 #resblock 2
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post3(self.pad_3Kernel_Res(z)))
#                 y = self.Conv_Post4(self.pad_3Kernel_Res(y))
#                 z = z + y 
#                 #resblock 3
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post5(self.pad_3Kernel_Res(z)))
#                 y = self.Conv_Post6(self.pad_3Kernel_Res(y))
#                 z = z + y      
#                 z = self.relu(z)
#                 z = self.relu(self.Conv4(z))
#                 z = self.relu(self.Conv5(z))
#                 z=self.ConvOut(z)
#                 return z.squeeze(1) 

#         # Forward function with 2 extra resblock at end
#         if (DataSetSize >=Factor*4):
#                 z = self.ConvInit1(self.pad_7Kernel(x2))
#                 z1 =  self.relu(self.ConvInit2(self.pad_5Kernel(z)))
#                 z2 =  self.AVG(z1)
#                 z3 =  self.AVG(z2)
#                 z4 =  self.AVG(z3)
#                 # Full scale
#                 z1 = self.relu(self.Conv11(self.pad_5Kernel(z1)))
#                 z1 = self.relu(self.Conv12(self.pad_5Kernel(z1)))
#                 # Downsample1 scale
#                 z2 = self.relu(self.Conv21(self.pad_5Kernel(z2)))
#                 z2 = self.relu(self.Conv22(self.pad_5Kernel(z2)))
#                 z2 =  self.Upsample(z2)
#                 # Downsample2 scale
#                 z3 = self.relu(self.Conv31(self.pad_3Kernel_3scale(z3)))
#                 z3 = self.relu(self.Conv32(self.pad_3Kernel_3scale(z3)))
#                 z3 = self.Upsample(z3)
#                 # Downsample3 scale
#                 z4 = self.relu(self.Conv41(( self.pad_3Kernel_4scale(z4))))
#                 z4 = self.relu(self.Conv42(( self.pad_3Kernel_4scale(z4))))
#                 z4 = self.Upsample(z4)
#                 # Sum all convolution output scales
#                 z = z1+z2+z3+z4
#                 #resblock 1
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post1(self.pad_3Kernel_Res(z)))
#                 y = self.Conv_Post2(self.pad_3Kernel_Res(y))
#                 z = z + y   
#                 z = self.relu(z)
#                 #resblock 2
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post3(self.pad_3Kernel_Res(z)))
#                 y = self.Conv_Post4(self.pad_3Kernel_Res(y))
#                 z = z + y 
#                 #resblock 3
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post5(self.pad_3Kernel_Res(z)))
#                 y = self.Conv_Post6(self.pad_3Kernel_Res(y))
#                 z = z + y     
#                 #resblock 4
#                 y = self.relu(z)
#                 y = self.relu(self.Conv_Post7(self.pad_3Kernel_Res(z)))
#                 y = self.Conv_Post8(self.pad_3Kernel_Res(y))
#                 z = z + y      
#                 z = self.relu(z)
#                 z = self.relu(self.Conv4(z))
#                 z = self.relu(self.Conv5(z))
#                 z=self.ConvOut(z)
#                 return z.squeeze(1) 




################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
## Unused network architecture code snipets



########################################################################
########################################################################
#__________________________________________________________________

# Resnet
#__________________________________________________________________
########################################################################
########################################################################
########################################################################
## RESNET

        # # 10 dim resolution
        # self.pad3   = torch.nn.ReflectionPad2d(3)
        # self.pad2   = torch.nn.ReflectionPad2d(2)
        # self.pad1   = torch.nn.ReflectionPad2d(1)



        # self.Conv1  = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7,bias=False)

        # self.Conv2  = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5,bias=False)
        # self.Conv3  = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5,bias=False)

        # self.Conv4  = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3,bias=False)
        # self.Conv5  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False)


        # self.Conv6  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False)
        # self.Conv7  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False)

        # self.Conv8  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False)
        # self.Conv9  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False)

        # self.Conv10  = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)


        # self.ConvSkip1 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1,bias=False)
        # self.ConvSkip2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1,bias=False)

        # self.relu   = torch.nn.LeakyReLU()

        ##_____________________________________________________________________________________________


        # 20 dim resolution
        # self.pad3   = torch.nn.ReflectionPad2d(6)
        # self.pad2   = torch.nn.ReflectionPad2d(4)
        # self.pad1   = torch.nn.ReflectionPad2d(2)

        # self.Conv1  = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7,bias=False,dilation=2)

        # self.Conv2  = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5,bias=False,dilation=2)
        # self.Conv3  = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5,bias=False,dilation=2)

        # self.Conv4  = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3,bias=False,dilation=2)
        # self.Conv5  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False,dilation=2)


        # self.Conv6  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False,dilation=2)
        # self.Conv7  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False,dilation=2)

        # self.Conv8  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False,dilation=2)
        # self.Conv9  = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,bias=False,dilation=2)

        # self.Conv10  = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)


        # self.ConvSkip1 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1,bias=False)
        # self.ConvSkip2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1,bias=False)

        # self.relu   = torch.nn.LeakyReLU()



#__________________________________________________________________
#     def forward(self, x,DataSetSize,Factor):
#         x2=x.unsqueeze(1)  # Add channel dimension (C) to input
#         Current_batchsize=int(x.shape[0])  # N in pytorch docs



#         #input layer
#         z = self.Conv1(self.pad3(x2))

# #       block 1
#         y = self.relu(z)
#         y = self.relu(self.Conv2(self.pad2(z)))
#         y = self.Conv3(self.pad2(y))
#         z = self.ConvSkip1(z)
#         z = z + y
#         z=self.relu(z)

        
# #       block 2
#         y = self.relu(z)
#         y = self.relu(self.Conv4(self.pad1(z)))
#         y = self.Conv5(self.pad1(y))
#         z = self.ConvSkip2(z)
#         z = z + y
#         z=self.relu(z)


#         #       block 3
#         if (DataSetSize >Factor*1):
#                 y = self.relu(z)
#                 y = self.relu(self.Conv6(self.pad1(z)))
#                 y = self.Conv7(self.pad1(y))
#                 z = z + y
#                 z=self.relu(z)


#         #       block 4
#         if (DataSetSize >Factor*2):
#                 y = self.relu(z)
#                 y = self.relu(self.Conv8(self.pad1(z)))
#                 y = self.Conv9(self.pad1(y))
#                 z = z + y
#                 z=self.relu(z)


# #       Consolidating convolution and output
#         z = self.Conv10(z)

#         return z.squeeze(1) 


########################################################################
########################################################################
#__________________________________________________________________
# Sequential- sequential CNN
#__________________________________________________________________
########################################################################
########################################################################
## ALL CNN
# #         10x10 , and 15x15 dim input
#         self.Conv1   = torch.nn.Conv2d(1,16,(9,9), stride=1, padding=(4,4), dilation=1, groups=1, bias=False, padding_mode='replicate')#even dim
#         self.Conv2   = torch.nn.Conv2d(16,16,(7,7),stride=1, padding=(3,3), dilation=1, groups=1, bias=False, padding_mode='replicate')#even dim
#         self.Conv3   = torch.nn.Conv2d(16,8,(5,5),stride=1, padding=(2,2), dilation=1, groups=1, bias=False, padding_mode='replicate')#even dim
#         self.Conv4   = torch.nn.Conv2d(8,4,(3,3),stride=1, padding=(1,1), dilation=1, groups=1, bias=False, padding_mode='replicate')#even dim
#         self.Conv5   = torch.nn.Conv2d(4,2,(3,3), stride=1, padding=(1,1), dilation=1, groups=1, bias=False, padding_mode='replicate')#even dim
#         self.Conv51   = torch.nn.Conv2d(2,2,(3,3), stride=1, padding=(1,1), dilation=1, groups=1, bias=False, padding_mode='replicate')#even dim
#         self.Conv52   = torch.nn.Conv2d(2,2,(3,3), stride=1, padding=(1,1), dilation=1, groups=1, bias=False, padding_mode='replicate')#even dim
#         self.Conv53   = torch.nn.Conv2d(2,2,(3,3), stride=1, padding=(1,1), dilation=1, groups=1, bias=False, padding_mode='replicate')#even dim
#         self.Conv54   = torch.nn.Conv2d(2,2,(3,3), stride=1, padding=(1,1), dilation=1, groups=1, bias=False, padding_mode='replicate')#even dim
#         self.Conv6   = torch.nn.Conv2d(2,1,(1,1), stride=1, padding=(0,0), dilation=1, groups=1, bias=False, padding_mode='replicate')#even dim

        
# #         #20x20 and 25x25 dim input
#         # self.Conv1   = torch.nn.Conv2d(1,16,(9,9), stride=1, padding=(8,8), dilation=2, groups=1, bias=True, padding_mode='reflect')#even dim
#         # self.Conv2   = torch.nn.Conv2d(16,16,(7,7),stride=1, padding=(6,6), dilation=2, groups=1, bias=True, padding_mode='reflect')#even dim
#         # self.Conv3   = torch.nn.Conv2d(16,8,(5,5),stride=1, padding=(4,4), dilation=2, groups=1, bias=True, padding_mode='reflect')#even dim
#         # self.Conv4   = torch.nn.Conv2d(8,4,(3,3),stride=1, padding=(1,1), dilation=1, groups=1, bias=True, padding_mode='reflect')#even dim
#         # self.Conv5   = torch.nn.Conv2d(4,2,(3,3), stride=1, padding=(1,1), dilation=1, groups=1, bias=True, padding_mode='reflect')#even dim
#         # self.Conv6   = torch.nn.Conv2d(2,1,(1,1), stride=1, padding=(0,0), dilation=1, groups=1, bias=True, padding_mode='reflect')#even dim

#         #  30x30 dim input
# #         self.Conv1   = torch.nn.Conv2d(1,16,(9,9), stride=1, padding=(16,16), dilation=4, groups=1, bias=True, padding_mode='reflect')#even dim
# #         self.Conv2   = torch.nn.Conv2d(16,16,(7,7),stride=1, padding=(12,12), dilation=4, groups=1, bias=True, padding_mode='reflect')#even dim
# #         self.Conv3   = torch.nn.Conv2d(16,8,(5,5),stride=1, padding=(8,8), dilation=4, groups=1, bias=True, padding_mode='reflect')#even dim
# #         self.Conv4   = torch.nn.Conv2d(8,4,(3,3),stride=1, padding=(3,3), dilation=3, groups=1, bias=True, padding_mode='reflect')#even dim
# #         self.Conv5   = torch.nn.Conv2d(4,2,(3,3), stride=1, padding=(3,3), dilation=3, groups=1, bias=True, padding_mode='reflect')#even dim
# #         self.Conv6   = torch.nn.Conv2d(2,1,(1,1), stride=1, padding=(0,0), dilation=1, groups=1, bias=True, padding_mode='reflect')#even dim

#         self.relu   = torch.nn.LeakyReLU()

#         self.BN1    = torch.nn.BatchNorm2d(16)
#         self.BN2    = torch.nn.BatchNorm2d(16)
#         self.BN3    = torch.nn.BatchNorm2d(8)


###___________________________________________________________________
#     def forward(self, x,DataSetSize):
#         """
#         In the forward function we accept a Tensor of input data and we must
#         return a Tensor of output data. We can use Modules defined in the
#         constructor as well as arbitrary operators on Tensors.
#         """
        # Factor=35
#         print(DataSetSize)
        # Current_batchsize=int(x.shape[0])  # N in pytorch docs


#         x2=x.unsqueeze(1)  # Add channel dimension (C) to input 
#         ConvOut=self.relu(self.BN1(self.Conv1(x2)))
#         ConvOut=self.relu(self.BN2(self.Conv2(ConvOut)))
#         ConvOut=self.relu(self.BN3(self.Conv3(ConvOut)))
#         ConvOut=self.relu(self.Conv4(ConvOut))
#         ConvOut=self.relu(self.Conv5(ConvOut))
#         ConvOut=self.relu(self.Conv51(ConvOut))
#         ConvOut=self.relu(self.Conv52(ConvOut))
#         if (DataSetSize >Factor*1):
#                 ConvOut=self.relu(self.Conv53(ConvOut))
#         if (DataSetSize >Factor*2):
#                 ConvOut=self.relu(self.Conv54(ConvOut))
#         if (DataSetSize >Factor*3):
#                 ConvOut=self.relu(self.Conv55(ConvOut))
#         # if (DataSetSize >Factor*4):
#         #         ConvOut=self.relu(self.Conv56(ConvOut))
#         ConvOut=(self.Conv6(ConvOut))
#         y_pred = ConvOut.squeeze(1) #Remove channel dimension
#         return y_pred
            