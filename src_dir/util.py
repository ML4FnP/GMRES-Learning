#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import functools
import time
import math

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
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        # print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value,run_time
    return wrapper_timer #  no "()" here, we need the object to 
                         #  be returned.




def Gauss_pdf(xArr,loc,sig):
    return np.exp(-0.5*((xArr-loc)/sig)**2.0)/(2*sig*np.sqrt(np.pi))

def Gauss_pdf_2D(xGrid,yGrid,xloc,yloc,sig):
    return np.exp(-0.5*(  ((xGrid-xloc)/sig)**2.0 +((yGrid-yloc)/sig)**2.0 ))/(2*np.pi*sig**2.0)


def moving_average(a, n) :
    if n<25 :
        Window=int(math.ceil(0.5*n))
    else :
        Window=25
    return np.sum(a[-Window-1:-1])/Window
