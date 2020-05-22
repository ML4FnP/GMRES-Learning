#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

import functools

import time

import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout
from torch.optim    import Adam, SGD

from src_dir.nn_collection import TwoLayerNet

from src_dir import resid,timer,moving_average

class NNPredictor(object):

    def __init__(self,D_in,H,H2,D_out):
        # N is batch size; D_in is input dimension;
        # H is hidden dimension; H2 is  second hidden layer dimension. 
        # D_out is output dimension.

        self.N=1
        self.D_in=D_in
        self.H=H   
        self.H2=H2  
        self.D_out=D_out


        # Construct our model by instantiating the class defined above
        self.model = TwoLayerNet(self.D_in, self.H,self.H2 ,self.D_out)

        # Construct our loss function and an Optimizer. The call to model.parameters()
        # in the SGD constructor will contain the learnable parameters of the two
        # nn.Linear modules which are members of the model.
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        self.x = torch.empty(0, self.D_in)
        self.y = torch.empty(0, self.D_out)
        self._is_trained = False

        # Diagnostic data => remove in production
        self.loss_val = list()

        # Number of training steps
        # self.n_steps = 1000
        self.n_steps = 3000000000


    @property
    def is_trained(self):
        return self._is_trained


    @is_trained.setter
    def is_trained(self, value):
        self._is_trained = value


    @property
    def counter(self):
        return self.x.size(0)

    def retrain(self):
        self.loss_val = list()  # clear loss val history
        for t in range(self.n_steps):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self.model(self.x)

            # Compute and print loss
            loss = self.criterion(y_pred, self.y)
            self.loss_val.append(loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.is_trained = True

    @timer
    def retrain_timed(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.loss_val = list()  # clear loss val history
        self.loss_val.append(10.0)
        t=0
        e1=1e-3
        while self.loss_val[-1]> e1 and t<self.n_steps:
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self.model(self.x.to(device))

            # Compute and print loss
            loss = self.criterion(y_pred, self.y.to(device))
            self.loss_val.append(loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('loss:',loss.item(),t)
            t=t+1

        print('Final loss:',loss.item(),t)
        # numparams=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print('parameters',numparams)
        self.is_trained = True





    def add(self, x, y):
        # TODO: don't use `torch.cat` in this incremental mode => will scale poorly
        # instead: use batched buffers
        self.x = torch.cat((self.x, torch.from_numpy(x).unsqueeze_(0).float()), 0)
        self.y = torch.cat((self.y, torch.from_numpy(y).unsqueeze_(0).float()), 0)

    def predict(self, x):
        return np.squeeze(
            self.model.forward(
                torch.from_numpy(x).unsqueeze_(0).float() # inputs need to be [[x_1, x_2, ...]] as floats
            ).detach().numpy() # outputs need to be numpy (non-grad => detach)
        ) # outputs need to be [y_1, y_2, ...]


    @timer
    def predict_timed(self, x):
        a1=torch.from_numpy(x).unsqueeze_(0).float()
        a2=np.squeeze(self.model.forward(a1).detach().cpu().numpy()) 
        #a2=np.squeeze(self.model.forward(a1).detach().numpy())     # cpu version, above line may work for cpu only... not sure. 
        return a2
# inputs need to be [[x_1, x_2, ...]] as floats
 # outputs need to be numpy (non-grad => detach)
# outputs need to be [y_1, y_2, ...]


def nn_preconditioner(retrain_freq=10, debug=False,InputDim=2,HiddenDim=100,HiddenDim2=100,OutputDim=2 ):
    def my_decorator(func):
        func.predictor    = NNPredictor(InputDim,HiddenDim,HiddenDim2,OutputDim)
        func.retrain_freq = retrain_freq
        func.debug        = debug
        func.InputDim     = InputDim
        func.HiddenDim    = HiddenDim
        func.HiddenDim2    = HiddenDim2
        func.OutputDim    = OutputDim

        @functools.wraps(func)
        def speedup_wrapper(*args, **kwargs):

            A, b, x0, e, nmax_iter,IterErrList,IterErr0_AVG,ProbCount,restart,debug,refine, *eargs = args

            IterErr0=0

            if func.predictor.is_trained and refine==False:
                pred_x0 = func.predictor.predict(b)
            else:
                pred_x0 = x0

            target  = func(A, b, pred_x0, e, nmax_iter,IterErrList,IterErr0_AVG,ProbCount,restart,debug,refine, *eargs)

            res = target[-1]



            if refine==False :
                IterErr = resid(A, target, b)
                IterErr0=IterErr[0]
                IterErrList.append(IterErr0)
                if ProbCount>10 :
                    IterErr0_AVG=moving_average(np.asarray(IterErrList),ProbCount)
                    print(IterErr0,IterErr0_AVG)

            if IterErrList[-1] > IterErr0_AVG and refine==True and ProbCount>10:  # Adhoc condition on residual of step to avoid overfitting. Approach doesn't seem to do better than this.
                func.predictor.add(b, res)
                if func.predictor.counter%retrain_freq== 0:
                    if func.debug:
                        print("retraining")
                        print(func.predictor.counter)
                        func.predictor.retrain()

            return target,IterErrList,IterErr0_AVG

        return speedup_wrapper
    return my_decorator




def nn_preconditioner_timed(retrain_freq=10, debug=False,InputDim=2,HiddenDim=100,HiddenDim2=100,OutputDim=2 ):
    def my_decorator(func):
        func.predictor    = NNPredictor(InputDim,HiddenDim,HiddenDim2,OutputDim)
        func.retrain_freq = retrain_freq
        func.debug        = debug
        func.InputDim     = InputDim
        func.HiddenDim    = HiddenDim
        func.HiddenDim2    = HiddenDim2
        func.OutputDim    = OutputDim

        @functools.wraps(func)
        def speedup_wrapper(*args, **kwargs):

            A, b, x0, e, nmax_iter,IterErrList,ProbCount,restart,debug,refine,blist,reslist,IterErrList10,GmresRunTime, *eargs = args

            forwardTime=0.0
            trainTime=0.0
            IterTime=0
            Initial_set=3

            IterTime_AVG=0.0
            IterErr10_AVG=0.0
            IterErrRatio=0.0

            if ProbCount>Initial_set :
                IterErr10_AVG=moving_average(np.asarray(IterErrList10),ProbCount)


            if func.predictor.is_trained and refine==False:
                pred_x0,forwardTime = func.predictor.predict_timed(b)
            else:
                pred_x0 = x0


            tic = time.perf_counter()
            target  = func(A, b, pred_x0, e, nmax_iter,IterErrList,ProbCount,restart,debug,refine,blist,reslist,IterErrList10,GmresRunTime, *eargs)
            toc = time.perf_counter()

            res = target[-1]



            if refine==False :
                IterErr = resid(A, target, b)
                IterTime=(toc-tic)
                IterErr10=IterErr[10]
                IterErrList.append(IterTime)
                IterErrList10.append(IterErr10)
                if ProbCount>Initial_set :
                    IterTime_AVG=moving_average(np.asarray(IterErrList),ProbCount)
                    IterErr10_AVG=moving_average(np.asarray(IterErrList10),ProbCount)
                    print(IterTime,IterTime_AVG,IterErr[10],IterErr10_AVG)
                if ProbCount<=Initial_set:
                    func.predictor.add(b, res)
                if ProbCount==Initial_set:
                    func.predictor.add(b, res)
                    timeLoop=func.predictor.retrain_timed()
                    print('Initial Training')


            if ProbCount>Initial_set:
                IterTime_AVG=moving_average(np.asarray(IterErrList),ProbCount)
                IterErr10_AVG=moving_average(np.asarray(IterErrList10),ProbCount)



            if (  IterErrList10[-1] > IterErr10_AVG and IterErrList[-1]>IterTime_AVG) and  refine==True and ProbCount>Initial_set   :  # Adhoc condition on residual of step to avoid overfitting. Approach doesn't seem to do better than this.
                blist.append(b)
                reslist.append(res)
                if   len(blist)==3 :
                    resMat=np.asarray(reslist)
                    resMat_square=resMat**2
                    row_sums = resMat_square.sum(axis=1,keepdims=True)
                    resMat= resMat/np.sqrt(row_sums)
                    InnerProd=np.dot(resMat,resMat.T)
                    print('InnerProd',InnerProd)
                    func.predictor.add(np.asarray(blist)[0], np.asarray(reslist)[0])
                    cutoff=0.8
                    if InnerProd[0,1] and InnerProd[0,2]<cutoff :
                        if InnerProd[1,2]<cutoff :
                            func.predictor.add(np.asarray(blist)[1], np.asarray(reslist)[1])
                            func.predictor.add(np.asarray(blist)[2], np.asarray(reslist)[2])
                        elif InnerProd[1,2]>=cutoff: 
                            func.predictor.add(np.asarray(blist)[1], np.asarray(reslist)[1])
                    elif InnerProd[0,1]<cutoff :
                        func.predictor.add(np.asarray(blist)[1], np.asarray(reslist)[1])
                    elif InnerProd[0,2]<cutoff :
                        func.predictor.add(np.asarray(blist)[2], np.asarray(reslist)[2])
                    
                    if func.predictor.counter%retrain_freq== 0:
                        if func.debug:
                            print("retraining")
                            print(func.predictor.counter)
                            # func.predictor.retrain()
                            timeLoop=func.predictor.retrain_timed()
                            trainTime=float(timeLoop[-1])
                            blist=[]
                            reslist=[]
            return target,IterErrList,IterTime_AVG,trainTime,forwardTime,blist,reslist,IterErrList10,GmresRunTime

        return speedup_wrapper
    return my_decorator