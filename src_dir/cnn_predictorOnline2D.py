#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

from functools import wraps
from inspect   import getfullargspec, signature
from copy      import deepcopy

import time

import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout

from torch.optim    import Adam, SGD

# from src_dir.cnn_collectionOnline2D import CnnOnline_2D

from src_dir import resid,timer,moving_average,GMRES

from src_dir import StatusPrinter



class CNNPredictorOnline_2D(object):

    def __init__(self, D_in, D_out, Area, dx, Model):

        # N is batch size; D_in is input dimension;
        # D_out is output dimension.
        self.D_in=D_in
        self.D_out=D_out

        ## Domain area and finite difference stencil width
        self.Area=Area
        self.dx=dx

        ## Increase layer at every multiple of this factor
        self.Factor=40

        ## Set Pytorch Seed
        torch.manual_seed(0)

        ## Construct our model by instantiating the class defined above
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model = CnnOnline_2D(self.D_in,self.D_out).to(device)
        self.model = Model(self.D_in, self.D_out).to(device)

        ## Construct our loss function and an Optimizer. The call to model.parameters()
        ## in the SGD constructor will contain the learnable parameters of the two
        ## nn.Conv modules which are members of the model.
        self.criterion = torch.nn.MSELoss(reduction='mean')

        ### Set optimizer
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        ## x will hold entire training set b data
        ## y will hold entire training set solution data
        self.x = torch.empty(0, self.D_in,self.D_in).to(device)
        self.y = torch.empty(0, self.D_out,self.D_out).to(device)

        # xNew holds new b additions to training set at the current time
        # yNew holds new solution (x) additions to training set at the current time
        self.xNew = torch.empty(0, self.D_in,self.D_in)
        self.yNew = torch.empty(0, self.D_out,self.D_out)

        # Set train flag
        self._is_trained = False

        # Diagnostic data => remove in production
        self.loss_val = list()


    @property
    def is_trained(self):
        return self._is_trained


    @is_trained.setter
    def is_trained(self, value):
        self._is_trained = value


    @property
    def counter(self):
        # Counter is based off of the data set to be added to training set 
        return self.xNew.size(0)


    @timer
    def retrain_timed(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.xNew = self.xNew.to(device)
        self.yNew = self.yNew.to(device)

        self.x=torch.cat((self.x,self.xNew))
        self.y=torch.cat((self.y,self.yNew))

        self.loss_val = list()  # clear loss val history
        self.loss_val.append(10.0)

        batch_size=16
        numEpochs=1000
        e1=1e-15
        epoch=0

        while self.loss_val[-1]> e1 and epoch<numEpochs-1:
            permutation = torch.randperm(self.x.size()[0])
            for t in range(0,self.x.size()[0],batch_size):

                ## indicies of random batch
                indices = permutation[t:t+batch_size]

                ## dataset batches
                batch_x, batch_y = self.x[indices],self.y[indices]

                ## batch of predictions
                y_pred = self.model(batch_x,self.x.size(0),self.Factor)

                ## Compute and print loss
                loss = (self.criterion(y_pred, batch_y))
                self.loss_val.append(loss.item())

                ## Print loss to console
                StatusPrinter().update_training(loss.item())

                ## Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch=epoch+1

        ## Add recent data to final batch and take one more step:

        permutation = torch.randperm(self.x.size()[0])
        indices = permutation[0:0+batch_size]
        batch_x, batch_y = self.x[indices],self.y[indices]

        # Adding new data to each batch
        # Note: only adding at most 3 data points to each batch
        batch_xMix=torch.cat((batch_x,self.xNew))
        batch_yMix=torch.cat((batch_y,self.yNew))

        ## Forward pass: Compute predicted y by passing x to the model
        y_pred = self.model(batch_xMix,self.x.size(0),self.Factor)

        ## Compute and print loss
        loss = (self.criterion(y_pred, batch_yMix))
        self.loss_val.append(loss.item())

        ## Print loss to console
        StatusPrinter().update_training(loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ## Clear tensors that are used to add data to training set
        self.xNew = torch.empty(0, self.D_in,self.D_in)
        self.yNew = torch.empty(0, self.D_out,self.D_out)

        ## Print number of parameters to console
        numparams=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        StatusPrinter().update_training_summary(numparams, self.x.size(0))

        self.is_trained = True


    def add(self, x, y):
        # TODO: don't use `torch.cat` in this incremental mode => will scale poorly
        # instead: use batched buffers
        self.xNew = torch.cat((self.xNew, torch.from_numpy(x).unsqueeze_(0).float()), 0)
        self.yNew = torch.cat((self.yNew, torch.from_numpy(y).unsqueeze_(0).float()), 0)

    def add_init(self, x, y):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.x = torch.cat((self.x, torch.from_numpy(x).unsqueeze_(0).float().to(device)), 0)
        self.y = torch.cat((self.y, torch.from_numpy(y).unsqueeze_(0).float().to(device)), 0)


    def predict(self, x):
        # inputs need to be [[x_1, x_2, ...]] as floats
        # outputs need to be numpy (non-grad => detach)
        # outputs need to be [y_1, y_2, ...]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        a1=torch.from_numpy(x).unsqueeze_(0).float().to(device)
        a2=np.squeeze(self.model.forward(a1,self.x.size(0),self.Factor).detach().cpu().numpy())
        return a2



class ArgsView(object):
    """
    A view class that gives access to input arguments
    """

    def __init__(self, spec, args, kwargs):
        self.__spec = spec
        self.__args = args
        self.__kwargs = kwargs

        for i, arg_name in enumerate(self.__spec.args):
            setattr(
                self,
                arg_name,
                self.__arg_view(i, arg_name)
            )

        self._vargs = []
        for j in range(i+1, len(self.__args)):
            self._vargs.append(
                self.__varg_view(j)
            )

        for arg_name in self.__spec.kwonlyargs:
            setattr(
                self,
                arg_name,
                self.__kwoarg_view(arg_name)
            )


    @property
    def _spec(self):
        return self.__spec


    def _replace(self, name, data):
        setattr(self, name, lambda: data)


    def __arg_view(self, arg_idx, arg_name):
        if arg_idx < len(self.__args):
            return lambda: self.__args[arg_idx]
        if arg_name in self.__kwargs:
            return lambda: self.__kwargs[arg_name]

        defaults_offset = len(self.__spec.args) - len(self.__spec.defaults)
        return lambda: self.__spec.defaults[arg_idx - defaults_offset]


    def __kwoarg_view(self, arg_name):
        if arg_name in self.__kwargs:
            return lambda: self.__kwargs[arg_name]

        return lambda: self.__spec.kwonlydefaults[arg_name]


    def __varg_view(self, idx):
        return lambda: self.__args[idx]


    def __str__(self):
        return str(self.__dict__.keys())


    def __repr__(self):
        return repr(self.__dict__.keys())



class PreconditionerTrainer(object):

    def __init__(
            self, preconditioner, linop_name="A", prob_rhs_name="b",
            prob_lhs_name="x", prob_init_name="x0", prob_tolerance_name="e",
            retrain_freq=1, debug=False, Initial_set=32
        ):

        self.preconditioner = preconditioner
        self.retrain_freq   = retrain_freq
        self.debug          = debug
        self.Initial_set    = Initial_set

        # Describe how we get specific arguments out of the input args
        self.linop_name          = linop_name
        self.prob_rhs_name       = prob_rhs_name
        self.prob_lhs_name       = prob_lhs_name
        self.prob_init_name      = prob_init_name
        self.prob_tolerance_name = prob_tolerance_name

        self.ML_GMRES_Time_list = list()
        self.ProbCount          = 0
        self.prob_debug         = False,
        self.blist              = list()
        self.reslist            = list()
        self.Err_list           = list()
        self.reslist_flat       = list()
        self.IterErrList        = list()


    def set_args_view(self, spec, args, kwargs):
        self.args_view = ArgsView(spec, args, kwargs)


    def get_problem_data(self):
        # Construct view into A, x, b, and x0
        A  = getattr(self.arg_view, trainer.linop_name)
        b  = getattr(self.arg_view, trainer.prob_rhs_name)
        x0 = getattr(self.arg_view, trainer.prob_init_name)
        e  = getattr(self.arg_view, trainer.prob_tolerance_name)

        return A(), b(), x0(), e()


    @staticmethod
    def fill_args(arg_view):
        args = []
        for arg in arg_view._spec.args:
            args.append(
                getattr(arg_view, arg)()
            )

        for argv in arg_view._vargs:
            args.append(
                argv()
            )

        kwargs = dict()
        for kw in arg_view._spec.kwonlyargs:
            kwargs[kw] = getattr(arg_view, kw)()

        return args, kwargs



def cnn_preconditionerOnline_timed_2D(trainer):

    def my_decorator(func):
        # func.predictor    = CNNPredictorOnline_2D(InputDim,OutputDim,Area,dx)
        spec = getfullargspec(func)
        name = func.__name__

        @wraps(func)
        def speedup_wrapper(*args, **kwargs):

            # Construct view into A, x, b, and x0
            trainer.set_args_view(spec, args, kwargs)

            # Get problem data:
            A, b, x0, e = trainer.get_problem_data()

            # Initialize NN total train time for iteration with zero value
            trainTime=0.0

            ## Compute 2-norm of RHS(for scaling RHS input to network)
            b_flat=np.reshape(b,(1,-1),order='F').squeeze(0)
            b_norm=np.linalg.norm(b_flat)
            b_Norm_max= np.max(b/b_norm)

            if trainer.preconditioner.is_trained:
                pred_x0 = trainer.preconditioner.predict(b/b_norm/b_Norm_max)
                pred_x0 = pred_x0*b_norm*b_Norm_max
                # target_test=GMRES(A, b, x0, e, 6,1, True)
                # IterErr_test = resid(A, target_test, b)
                # print('size',len(IterErr_test))
                # print(IterErr_test[5],max(trainer.Err_list))
                # if (IterErr_test[5]>1.75*max(trainer.Err_list)):
                #     print('poor prediction,using initial x0')
                # pred_x0 = x0
            else:
                pred_x0 = x0


            # Replace the input initial guess witht the NN preconditioner
            args_view = deepcopy(trainer.args_view)
            args_view._replace(trainer.prob_init_name, pred_x0)
            new_args, new_kwargs = PreconditionerTrainer.fill_args(args_view)

            ## Time GMRES function
            tic = time.perf_counter()
            # target  = func(A, b, pred_x0, e, ML_GMRES_Time_list,ProbCount,debug,blist,reslist,Err_list,reslist_flat,IterErrList, *eargs)
            target = func(*new_args, *new_kwargs)
            toc = time.perf_counter()

            ## Pick out solution from residual list
            res = target[-1]

            ## Write diagnostic data (error and time-to solution) to list
            IterErr = resid(A, target, b)
            trainer.IterErrList.append(IterErr)
            IterTime  = (toc-tic)
            IterErr10 = IterErr[22]
            trainer.ML_GMRES_Time_list.append(IterTime)
            trainer.Err_list.append(IterErr10)

            ## Rescale RHS so that network is trained on normalized data
            b=b/b_norm/b_Norm_max
            res=res/b_norm/b_Norm_max


            if trainer.ProbCount <= trainer.Initial_set:
                trainer.preconditioner.add_init(b,res)
            if trainer.ProbCount == trainer.Initial_set:
                timeLoop=trainer.preconditioner.retrain_timed()
                # print('Initial Training')

            ## Compute moving averages used to filter data
            if trainer.ProbCount > trainer.Initial_set:
                IterTime_AVG = moving_average(
                        np.asarray(trainer.ML_GMRES_Time_list),
                        trainer.ProbCount
                    )
                IterErr10_AVG = moving_average(
                        np.asarray(trainer.Err_list),
                        trainer.ProbCount
                    )
                # print(trianer.ML_GMRES_Time_list[-1],IterTime_AVG,trainer.Err_list[-1],IterErr10_AVG)


            ## Filter for data to be added to training set
            if trainer.ProbCount > trainer.Initial_set:
                if trainer.ML_GMRES_Time_list[-1]>IterTime_AVG and trainer.Err_list[-1]>IterErr10_AVG:

                    CoinToss=np.random.rand()
                    if (CoinToss < 0.5):
                        trainer.blist.append(b)
                        trainer.reslist.append(res)
                        trainer.reslist_flat.append(
                                np.reshape(res,(1,-1), order='C').squeeze(0)
                            )

                    ## check orthogonality of 3 solutions that met training set critera
                    if len(trainer.blist) == 3:
                        resMat        = np.asarray(trainer.reslist_flat)
                        resMat_square = resMat**2
                        row_sums      = resMat_square.sum(axis=1, keepdims=True)
                        resMat        = resMat/np.sqrt(row_sums)
                        InnerProd     = np.dot(resMat, resMat.T)
                        # print('InnerProd',InnerProd)

                        #TODO: Do we need np.asarray here?
                        trainer.preconditioner.add(
                                np.asarray(trianer.blist)[0],
                                np.asarray(trainer.reslist)[0]
                            )

                        cutoff=0.8
                        ## Picking out sufficiently orthogonal subset of 3 solutions gathered
                        if np.abs(InnerProd[0,1]) and np.abs(InnerProd[0,2])<cutoff:
                            if np.abs(InnerProd[1,2]) < cutoff:

                                #TODO: Do we need np.asarray here?
                                trainer.preconditioner.add(
                                        np.asarray(trainer.blist)[1],
                                        np.asarray(trainer.reslist)[1]
                                    )

                                #TODO: Do we need np.asarray here?
                                trainer.preconditioner.add(
                                        np.asarray(trainer.blist)[2],
                                        np.asarray(trainer.reslist)[2]
                                    )

                            elif np.abs(InnerProd[1,2]) >= cutoff:
                                #TODO: Do we need np.asarray here?
                                trainer.preconditioner.add(
                                        np.asarray(trainer.blist)[1],
                                        np.asarray(trainer.reslist)[1]
                                    )

                        elif np.abs(InnerProd[0,1]) < cutoff :
                            #TODO: Do we need np.asarray here?
                            trainer.preconditioner.add(
                                    np.asarray(trainer.blist)[1],
                                    np.asarray(trainer.reslist)[1]
                                )

                        elif np.abs(InnerProd[0,2]) < cutoff :
                            #TODO: Do we need np.asarray here?
                            trainer.preconditioner.add(
                                    np.asarray(trainer.blist)[2],
                                    np.asarray(trainer.reslist)[2]
                                )

                        ## Train if enough data has been collected
                        if trainer.preconditioner.counter>=trainer.retrain_freq:
                            # if trainer.debug:
                            #     print("retraining")
                            #     print(trainer.preconditioner.counter)
                            timeLoop=trainer.preconditioner.retrain_timed()
                            trainTime=float(timeLoop[-1])
                            trainer.blist        = []
                            trainer.reslist      = []
                            trainer.reslist_flat = []
            return target# ,ML_GMRES_Time_list,trainTime,blist,reslist,Err_list,reslist_flat,IterErrList

        speedup_wrapper.__signature__ = signature(func)
        return speedup_wrapper

    return my_decorator





##########################33
# Possibly useful snipets

# #original implementation of tensor linop A (prohibitively slow for back prop)
# self.ATesnorOp = mk_laplace_2d_Tensor(10, 10, dx)
# y_pred= self.ATesnorOp(y_pred)
# loss = self.criterion(y_pred-batch_x,0*y_pred)

 ## FD convolutional weights for computing residual
# self.FDpad=torch.nn.ZeroPad2d(1).to(device)
# self.Aweights = torch.tensor([[0., 1., 0.],
#                 [1., -4., 1.],
#                 [0.,  1., 0.]])*(1/dx)**2.0
# self.Aweights = self.Aweights.to(device)
# self.Aweights = self.Aweights.view(1,1,3 ,3 )


# #For restoring original scale of  solutions and RHS
# y_pred=torch.mul(y_pred,batch_Normfactors)
# batch_y =torch.mul(batch_y,batch_Normfactors)
# batch_x =torch.mul(batch_x,batch_Normfactors)
# ResidualLoss = torch.mul(ResidualLoss,batch_Normfactors)

# #Code for writing number of samples to file
# f2=open("NumSamples.txt","ab")
# Temp=np.zeros((1,1))
# Temp[0,0]=self.x.size(0)
# np.savetxt(f2,Temp)
# f2.close()


## Code for writing loss values to files
# f=open("Losses.txt","ab")
# # print(np.asarray(self.loss_val[1:-1]))
# np.savetxt(f,np.asarray(self.loss_val[1:-1]))
# f.close()



# Alternate implementation of loss function
# ResidualLoss = torch.square(ResidualLoss)
# ResidualLoss = torch.sum(ResidualLoss,-1)
# ResidualLoss = torch.sum(ResidualLoss,-1)
# ResidualLoss = torch.sqrt(ResidualLoss)
# ResidualLoss = torch.sum(ResidualLoss)
# ResidualLoss = torch.sqrt(0.0001*ResidualLoss/y_pred.size(0))


# L2Integralloss = torch.square(y_pred-batch_y)
# L2Integralloss = torch.sum(L2Integralloss,-1)
# L2Integralloss = torch.sum(L2Integralloss,-1)
# L2Integralloss = torch.sqrt(L2Integralloss*(self.dx**2.0/self.Area))
# L2Integralloss = torch.sum(L2Integralloss)
# L2Integralloss =  torch.sqrt(L2Integralloss/y_pred.size(0))


# Faster implentation of loss
# ResidualLoss=torch.nn.functional.conv2d(self.FDpad(y_pred.unsqueeze(1)), self.Aweights, bias=None, stride=1)
# ResidualLoss = ResidualLoss.squeeze(1)
# ResidualLoss = ResidualLoss - batch_x
# ResidualLoss= (0.0001*self.criterion(ResidualLoss, 0.0*ResidualLoss))

# loss= torch.sqrt(L2Integralloss+ResidualLoss)



## snippets for restoring  RHS scale during training
# barray=np.ones((InputDim,InputDim))
# barray=barray*b_norm
# bnormList.append(barray)
