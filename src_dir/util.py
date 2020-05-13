#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import functools
import time

# define consistent linear math that stick with `np.array` (rather than
# `np.matrix`) => this will mean that we're sticking with the "minimal" data
# type for vector data. NOTE: this might cause a performance hit due to
# changing data types.
mat_to_a = lambda a    : np.squeeze(np.asarray(a))
matmul_a = lambda a, b : mat_to_a(np.dot(a, b))


def resid(A, x, b):
    return np.array(
        [np.linalg.norm(matmul_a(A, xi)-b) for xi in x]
    )


# mathematical indices for python
cidx   = lambda i: i-1  # c-style index from math-style index
midx   = lambda i: i+1  # math-style index from c-style index
mrange = lambda n: range(1, n + 1)


# 1D sclar laplace operator
def laplace_1d(N):
    op = np.matrix(np.zeros((N, N)))
    for i in mrange(N):
        if i > 1:
            op[cidx(i), cidx(i-1)] = -1
        op[cidx(i), cidx(i)] = 2
        if i < N:
            op[cidx(i), cidx(i+1)] = -1
    return op






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
    return np.exp(-0.5*((xArr-loc)/sig)**2.0)/(sig*np.sqrt(np.pi))