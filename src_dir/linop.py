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

    for ix, in np.ndindex(Ax.shape):
        Ax[ix] = op(x, ix+1)

    return Ax



def mk_laplace_1d(N, bc="dirichlet", lval=0, rval=0):
    '''
    mk_laplace_1d(N, bc="dirichlet", lval=0, rval=0)

    Generates laplace operator as a stencil operation for a N-cell 1D grid, for
    given boundary conditionds.
    '''
    
    def build_bc_1(x_in):
        x_out = np.zeros((N + 2,))

        x_out[1:N+1] = x_in[:]

        x_out[0]   = lval
        x_out[N+1] = rval

        return x_out

    
    def laplace_1d(x, i):
        return 2*x[i] - x[i-1] - x[i+1]


    if bc == "dirichlet":
        op  = lambda x, i: laplace_1d(build_bc_1(x), i)
    else:
        raise RuntimeError(f"bc={bc} not implemented")

    return lambda x: appl_1d(op, x)



def appl_2d(op, x, Nx, Ny):
    Ax = np.zeros((Nx, Ny))

    for ix, iy in np.ndindex(Ax.shape):
        Ax[ix, iy] = op(x, ix+1, iy+1)

    return Ax



def mk_laplace_2d(Nx, Ny, bc="dirichlet", xlo=0, xhi=0, ylo=0, yhi=0):
    '''
    mk_laplace_2d(N, bc="dirichlet", xlo=0, xhi=0, ylo=0, yhi=0)

    Generates laplace operator as a stencil operation for a N-cell 2D grid, for
    given boundary conditionds.
    '''

    def build_bc_1(x_in):
        x_out = np.zeros((Nx + 2, Ny + 2))

        x_out[1:Nx+1, 1:Ny+1] = x_in[:, :]

        x_out[0,    :] = xlo
        x_out[Nx+1, :] = xhi
        x_out[:,    0] = ylo
        x_out[:, Ny+1] = yhi

        return x_out


    def build_bc_2(x_in):
        x_out = np.zeros((Nx + 2, Ny + 2))

        x_out[1:Nx+1, 1:Ny+1] = x_in[:, :]

        x_out[0,    1:Nx+1] = x_in[-1, :]
        x_out[Nx+1, 1:Nx+1] = x_in[0,  :]
        x_out[1:Nx+1,    0] = x_in[:, -1]
        x_out[1:Nx+1, Ny+1] = x_in[:,  0]

        return x_out


    def laplace_2d(x, i, j):
        return -4*x[i, j] + x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1]


    if bc == "dirichlet":
        op  = lambda x, i, j: laplace_2d(build_bc_1(x), i, j)
    elif bc == "periodic":
        op  = lambda x, i, j: laplace_2d(build_bc_2(x), i, j)
    else:
        raise RuntimeError(f"bc={bc} not implemented")

    return lambda x: appl_2d(op, x, Nx, Ny)
