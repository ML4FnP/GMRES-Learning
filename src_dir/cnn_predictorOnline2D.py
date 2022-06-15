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

from src_dir import prob_norm, resid, timer, moving_average

from src_dir import StatusPrinter



class CNNPredictorOnline_2D(object):

    def __init__(self, D_in, D_out, Area, dx, Model):

        # N is batch size; D_in is input dimension;
        # D_out is output dimension.
        self.D_in  = D_in
        self.D_out = D_out

        # Domain area and finite difference stencil width
        self.Area = Area
        self.dx   = dx

        # Increase layer at every multiple of this factor
        self.Factor = 40

        # Set Pytorch Seed
        torch.manual_seed(0)

        # Construct our model by instantiating the class defined above
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model(self.D_in, self.D_out).to(device)

        # Construct our loss function and an Optimizer. The call to
        # model.parameters() in the SGD constructor will contain the learnable
        # parameters of the two nn.Conv modules which are members of the model.
        self.criterion = torch.nn.MSELoss(reduction='mean')

        ### Set optimizer
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # x will hold entire training set b data
        # y will hold entire training set solution data
        self.x = torch.empty(0, self.D_in,  self.D_in).to(device)
        self.y = torch.empty(0, self.D_out, self.D_out).to(device)

        # xNew: new b additions to training set at the current time
        # yNew: new solution (x) additions to training set at the current time
        self.xNew = torch.empty(0, self.D_in,  self.D_in)
        self.yNew = torch.empty(0, self.D_out, self.D_out)

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

        self.x = torch.cat((self.x, self.xNew))
        self.y = torch.cat((self.y, self.yNew))

        self.loss_val = list()  # clear loss val history
        self.loss_val.append(10.0)

        batch_size = 16
        numEpochs  = 1000
        e1         = 1e-15
        epoch      = 0

        while self.loss_val[-1] > e1 and epoch < numEpochs - 1:
            permutation = torch.randperm(self.x.size()[0])
            for t in range(0, self.x.size()[0], batch_size):

                ## indicies of random batch
                indices = permutation[t:t+batch_size]

                ## dataset batches
                batch_x, batch_y = self.x[indices], self.y[indices]

                ## batch of predictions
                y_pred = self.model(batch_x, self.x.size(0), self.Factor)

                ## Compute and print loss
                loss = (self.criterion(y_pred, batch_y))
                self.loss_val.append(loss.item())

                ## Print loss to console
                StatusPrinter().update_training(loss.item())

                ## Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch=epoch + 1

        ## Add recent data to final batch and take one more step:

        permutation = torch.randperm(self.x.size()[0])
        indices     = permutation[0:0 + batch_size]
        batch_x, batch_y = self.x[indices], self.y[indices]

        # Adding new data to each batch
        # Note: only adding at most 3 data points to each batch
        batch_xMix = torch.cat((batch_x, self.xNew))
        batch_yMix = torch.cat((batch_y, self.yNew))

        ## Forward pass: Compute predicted y by passing x to the model
        y_pred = self.model(batch_xMix, self.x.size(0), self.Factor)

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
        self.xNew = torch.empty(0, self.D_in,  self.D_in)
        self.yNew = torch.empty(0, self.D_out, self.D_out)

        ## Print number of parameters to console
        numparams = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        StatusPrinter().update_training_summary(numparams, self.x.size(0))

        self.is_trained = True


    def add(self, x, y):
        # TODO: don't use `torch.cat` in this incremental mode => will scale
        # poorly instead: use batched buffers
        self.xNew = torch.cat(
                (self.xNew, torch.from_numpy(x).unsqueeze_(0).float()), 0
            )
        self.yNew = torch.cat(
                (self.yNew, torch.from_numpy(y).unsqueeze_(0).float()), 0
            )


    def add_init(self, x, y):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.x = torch.cat(
                (self.x, torch.from_numpy(x).unsqueeze_(0).float().to(device)),
                0
            )
        self.y = torch.cat(
                (self.y, torch.from_numpy(y).unsqueeze_(0).float().to(device)),
                0
            )


    def predict(self, x):
        # inputs need to be [[x_1, x_2, ...]] as floats
        # outputs need to be numpy (non-grad => detach)
        # outputs need to be [y_1, y_2, ...]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        a1 = torch.from_numpy(x).unsqueeze_(0).float().to(device)
        a2 = np.squeeze(
                self.model.forward(
                    a1, self.x.size(0), self.Factor
                ).detach().cpu().numpy()
            )
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
            retrain_freq=1, debug=False, Initial_set=32, diagnostic_probe=22
        ):

        self.preconditioner   = preconditioner
        self.retrain_freq     = retrain_freq
        self.debug            = debug
        self.Initial_set      = Initial_set
        self.diagnostic_probe = diagnostic_probe

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
        self.trainTime          = list()


    def set_args_view(self, spec, args, kwargs):
        self.args_view = ArgsView(spec, args, kwargs)


    def get_problem_data(self):
        # Construct view into A, b, x0, and e
        A  = getattr(self.args_view, self.linop_name)
        b  = getattr(self.args_view, self.prob_rhs_name)
        x0 = getattr(self.args_view, self.prob_init_name)
        e  = getattr(self.args_view, self.prob_tolerance_name)

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


    def predict(self, A, b, x0, b_scale):
        if self.preconditioner.is_trained:
            pred_x0 = self.preconditioner.predict(b/b_scale)
            pred_x0 = pred_x0 * b_scale
            # target_test=GMRES(A, b, x0, e, 6,1, True)
            # IterErr_test = resid(A, target_test, b)
            # print('size',len(IterErr_test))
            # print(IterErr_test[5],max(self.Err_list))
            # if (IterErr_test[5]>1.75*max(self.Err_list)):
            #     print('poor prediction,using initial x0')
            # pred_x0 = x0
        else:
            pred_x0 = x0

        return pred_x0


    def add_single(self, res, b, scale):

        # Rescale RHS so that network is trained on normalized data
        b   = b   / scale
        res = res / scale

        if self.ProbCount <= self.Initial_set:
            self.preconditioner.add_init(b, res)
        if self.ProbCount == self.Initial_set:
            timeLoop = self.preconditioner.retrain_timed()

        # Compute moving averages used to filter data
        if self.ProbCount > self.Initial_set:
            IterTime_AVG = moving_average(
                    np.asarray(self.ML_GMRES_Time_list),
                    self.ProbCount
                )
            IterErr10_AVG = moving_average(
                    np.asarray(self.Err_list),
                    self.ProbCount
                )

        # Filter for data to be added to training set
        if self.ProbCount > self.Initial_set:
            if self.ML_GMRES_Time_list[-1] > IterTime_AVG \
            and self.Err_list[-1] > IterErr10_AVG:

                CoinToss = np.random.rand()
                if (CoinToss < 0.5):
                    self.blist.append(b)
                    self.reslist.append(res)
                    self.reslist_flat.append(
                            np.reshape(res,(1,-1), order='C').squeeze(0)
                        )

                # check orthogonality of 3 solutions that met training set
                # critera
                if len(self.blist) == 3:
                    resMat        = np.asarray(self.reslist_flat)
                    resMat_square = resMat**2
                    row_sums      = resMat_square.sum(axis=1, keepdims=True)
                    resMat        = resMat/np.sqrt(row_sums)
                    InnerProd     = np.dot(resMat, resMat.T)

                    #TODO: Do we need np.asarray here?
                    self.preconditioner.add(
                            np.asarray(self.blist)[0],
                            np.asarray(self.reslist)[0]
                        )

                    cutoff=0.8
                    # Picking out sufficiently orthogonal subset of 3 solutions
                    # gathered
                    if np.abs(InnerProd[0,1]) < cutoff \
                    and np.abs(InnerProd[0,2]) < cutoff:
                        if np.abs(InnerProd[1,2]) < cutoff:

                            #TODO: Do we need np.asarray here?
                            self.preconditioner.add(
                                    np.asarray(self.blist)[1],
                                    np.asarray(self.reslist)[1]
                                )

                            #TODO: Do we need np.asarray here?
                            self.preconditioner.add(
                                    np.asarray(self.blist)[2],
                                    np.asarray(self.reslist)[2]
                                )

                        elif np.abs(InnerProd[1,2]) >= cutoff:
                            #TODO: Do we need np.asarray here?
                            self.preconditioner.add(
                                    np.asarray(self.blist)[1],
                                    np.asarray(self.reslist)[1]
                                )

                    elif np.abs(InnerProd[0,1]) < cutoff :
                        #TODO: Do we need np.asarray here?
                        self.preconditioner.add(
                                np.asarray(self.blist)[1],
                                np.asarray(self.reslist)[1]
                            )

                    elif np.abs(InnerProd[0,2]) < cutoff :
                        #TODO: Do we need np.asarray here?
                        self.preconditioner.add(
                                np.asarray(self.blist)[2],
                                np.asarray(self.reslist)[2]
                            )

                    # Train if enough data has been collected
                    if self.preconditioner.counter >= self.retrain_freq:
                        # if self.debug:
                        #     print("retraining")
                        #     print(self.preconditioner.counter)
                        timeLoop = self.preconditioner.retrain_timed()
                        # trainTime=float(timeLoop[-1])
                        # TODO: we need a data retention policy for things
                        # like the train time history
                        self.trainTime.append(timeLoop[-1])
                        self.blist        = []
                        self.reslist      = []
                        self.reslist_flat = []


    def write_diagnostics(self, iter_time, A, target, b):
        iter_err = resid(A, target, b)
        self.IterErrList.append(iter_err)

        iter_err_probe = iter_err[self.diagnostic_probe]
        self.ML_GMRES_Time_list.append(iter_time)
        self.Err_list.append(iter_err_probe)



def cnn_preconditionerOnline_timed_2D(trainer):

    def my_decorator(func):
        spec = getfullargspec(func)
        name = func.__name__

        @wraps(func)
        def speedup_wrapper(*args, **kwargs):

            # Construct view into A, x, b, and x0
            trainer.set_args_view(spec, args, kwargs)

            # Get problem data:
            A, b, x0, e = trainer.get_problem_data()

            # Use predictor to generate initial guess:
            b_norm, b_Norm_max = prob_norm(b)
            pred_x0            = trainer.predict(A, b, x0, b_norm*b_Norm_max)

            # Replace the input initial guess witht the NN preconditioner
            args_view = deepcopy(trainer.args_view)
            args_view._replace(trainer.prob_init_name, pred_x0)
            new_args, new_kwargs = PreconditionerTrainer.fill_args(args_view)

            # Run function (and time it)
            tic = time.perf_counter()
            target = func(*new_args, **new_kwargs)
            toc = time.perf_counter()

            # Pick out solution from residual list
            res = target[-1]

            # Write diagnostic data (error and time-to solution) to list
            IterTime = (toc-tic)
            trainer.write_diagnostics(IterTime, A, target, b)

            # Add problem to the training set
            trainer.add_single(res, b, b_norm*b_Norm_max)

            return target

        speedup_wrapper.__signature__ = signature(func)

        return speedup_wrapper

    return my_decorator

