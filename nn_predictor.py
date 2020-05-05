#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

import functools

import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout
from torch.optim    import Adam, SGD

from nn_collection import TwoLayerNet



class NNPredictor(object):

    def __init__(self):
        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        self.N, self.D_in, self.H, self.D_out = 1, 2, 100, 2

        # Construct our model by instantiating the class defined above
        self.model = TwoLayerNet(self.D_in, self.H, self.D_out)

        # Construct our loss function and an Optimizer. The call to model.parameters()
        # in the SGD constructor will contain the learnable parameters of the two
        # nn.Linear modules which are members of the model.
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)

        self.x = torch.empty(0, self.D_in)
        self.y = torch.empty(0, self.D_out)
        self._is_trained = False

        # Diagnostic data => remove in production
        self.loss_val = list()

        # Number of training steps
        self.n_steps = 1000


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





# TODO: make the wrapper also keep track of A => the NN depends on A
def nn_preconditioner(retrain_freq=10, debug=False):

    def my_decorator(func):
        func.predictor    = NNPredictor()
        func.retrain_freq = retrain_freq
        func.debug        = debug

        @functools.wraps(func)
        def speedup_wrapper(*args, **kwargs):

            A, b, x0, e, nmax_iter, *eargs = args

            if func.predictor.is_trained:
                pred_x0 = func.predictor.predict(b)
            else:
                pred_x0 = x0

            target  = func(A, b, pred_x0, e, nmax_iter, *eargs)

            res = target[-1]

            func.predictor.add(b, res)

            if func.predictor.counter%retrain_freq == 0:
                if func.debug:
                    print("retraining")
                func.predictor.retrain()

            return target

        return speedup_wrapper
    return my_decorator
