#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

from .util import mrange, cidx



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




def appl_1d(op, x):
    Ax = np.zeros_like(x)

    for i in range(len(x)):
        Ax[i] = op(x, i)

    return Ax




def mk_laplace_1d(N, bc="dirichlet", lval=0, rval=0):
    '''
    mk_laplace_1d(N, bc="dirichlet", lval=0, rval=0)

    Generates laplace operator as a stencil operation for a N-cell 1D grid, for
    given boundary conditionds.
    '''
    def laplace_1d_bc1(x, i, l):
        if i == 0:
            return 2*x[i] - x[i+1] - lval
        if i == l-1:
            return 2*x[i] - x[i-1] - rval
        return 2*x[i] - x[i-1] - x[i+1]

    if bc == "dirichlet":
        op  = lambda x, i: laplace_1d_bc1(x, i, N)
    else:
        raise RuntimeError(f"bc={bc} not implemented")

    return lambda x: appl_1d(op, x)
