#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import math

from functools import wraps
from inspect   import signature

# define consistent linear math that stick with `np.array` (rather than
# `np.matrix`) => this will mean that we're sticking with the "minimal" data
# type for vector data. NOTE: this might cause a performance hit due to
# changing data types.
mat_to_a = lambda a    : np.squeeze(np.asarray(a))
matmul_a = lambda a, b : mat_to_a(np.dot(a, b))



def resid(A, *args, **kwargs):
    if isinstance(A, np.matrix):
        A_op = lambda x: matmul_a(A, x)
        return resid_kernel(A_op, *args, **kwargs)
    else:
        return resid_kernel(A, *args, **kwargs)



def resid_kernel(A, x, b):
    return np.array(
        [np.linalg.norm(A(xi)-b) for xi in x]
    )



# mathematical indices for python
cidx   = lambda i: i-1  # c-style index from math-style index
midx   = lambda i: i+1  # math-style index from c-style index
mrange = lambda n: range(1, n + 1)



def timer(func):
    """Print the runtime of the decorated function"""

    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        # print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value,run_time

    # preserve the original function signature
    wrapper_timer.__signature__ = signature(func)

    return wrapper_timer




def Gauss_pdf(xArr,loc,sig):
    return np.exp(-0.5*((xArr-loc)/sig)**2.0)/(2*sig*np.sqrt(np.pi))



def Gauss_pdf_2D(xGrid,yGrid,xloc,yloc,sig):
    return np.exp(-0.5*(  ((xGrid-xloc)/sig)**2.0 +((yGrid-yloc)/sig)**2.0 ))/(2*np.pi*sig**2.0)



def moving_average(a, n):
    if n == 0:
        return 0
    elif n < 25:
        Window=int(math.ceil(0.5*n))
    else:
        Window=25
    return np.sum(a[-Window-1:-1])/Window



class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]



# From here: https://stackoverflow.com/questions/61223812/python-print-how-to-clear-output-before-print-end-r
class OverwriteLast(object):

    def __init__(self):
        self.last = 0

    def print(self, s):
        if self.last:
            print(" "*self.last, end="\r")
        self.last = len(s)
        print(s, end="\r")

        

class StatusPrinter(object, metaclass=Singleton):
    
    def __init__(self):
        self._printer = OverwriteLast()
        self._loss    = 0
        self._speedup = 0
        self._iter    = 0
        self._num_p   = 0
        self._num_d   = 0
        

    def __str__(self):
        return (
            f"iter={self._iter} "
            f"speedup={self._speedup:0.4f} "
            f"loss={self._loss:0.4e} "
            f"parameters={self._num_p} "
            f"data size={self._num_d}"
        )

    
    def finalize(self):
        self.__init__()
        print("", flush=True)


    def print(self):
        # self._printer.print(f"iter={self._iter:<5} speedup={self._speedup:0.5f} loss={self._loss:0.5e} number of parameters={self._num_p} number of data points={self._num_d}")
        self._printer.print(str(self))


    def update_training(self, loss):
        self._loss = loss
        self.print()
    
    
    def update_simulation(self, speedup, idx):
        self._speedup = speedup
        self._iter    = idx
        self.print()
        

    def update_training_summary(self, num_parameters, num_data_points):
        self._num_p = num_parameters
        self._num_d = num_data_points
        self.print()
