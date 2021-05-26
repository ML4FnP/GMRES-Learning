#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from   numba import njit

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
        return (-4*x[i, j] + x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])


    if bc == "dirichlet":
        op  = lambda x, i, j: laplace_2d(build_bc_1(x), i, j)
    elif bc == "periodic":
        op  = lambda x, i, j: laplace_2d(build_bc_2(x), i, j)
    else:
        raise RuntimeError(f"bc={bc} not implemented")

    return lambda x: appl_2d(op, x, Nx, Ny)



@njit(nogil=True)
def laplace_2d(x, i, j):
    return (-4*x[i, j] + x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])



@njit(nogil=True)
def laplace_2d_dirichlet(x_in, xlo, xhi, ylo, yhi):
    '''
    laplace_2d_dirichlet(x_in, xlo, xhi, ylo, yhi)

    Applies laplace operator as a stencil operation for a N-cell 2D grid, for
    given boundary conditionds.
    '''

    Nx, Ny = x_in.shape
    x_pad = np.zeros((Nx + 2, Ny + 2))
    x_pad[1:Nx+1, 1:Ny+1] = x_in[:, :]

    x_pad[0,    :] = xlo
    x_pad[Nx+1, :] = xhi
    x_pad[:,    0] = ylo
    x_pad[:, Ny+1] = yhi

    Ax = np.zeros((Nx, Ny))
    for ix, iy in np.ndindex(Ax.shape):
        Ax[ix, iy] = laplace_2d(x_pad, ix+1, iy+1)

    return Ax



@njit(nogil=True)
def laplace_2d_extrap(x_in):
    '''
    laplace_2d_extrap(x_in)

    Applies laplace operator as a stencil operation for a N-cell 2D grid, for
    given boundary conditionds.
    '''

    Nx, Ny = x_in.shape
    x_pad = np.zeros((Nx + 2, Ny + 2))
    x_pad[1:Nx+1, 1:Ny+1] = x_in[:, :]

    x_pad[0,    :] = x_pad[1,  :] - (x_pad[2,  :] - x_pad[1,    :])
    x_pad[Nx+1, :] = x_pad[Nx, :] + (x_pad[Nx, :] - x_pad[Nx-1, :])
    x_pad[:,    0] = x_pad[:,  1] - (x_pad[:,  2] - x_pad[:,    1])
    x_pad[:, Ny+1] = x_pad[:, Ny] + (x_pad[:, Ny] - x_pad[:, Ny-1])

    Ax = np.zeros((Nx, Ny))
    for ix, iy in np.ndindex(Ax.shape):
        Ax[ix, iy] = laplace_2d(x_pad, ix+1, iy+1)

    return Ax



@njit(nogil=True)
def laplace_2d_constextrap(x_in):
    '''
    laplace_2d_constextrap(x_in)

    Applies laplace operator as a stencil operation for a N-cell 2D grid, for
    given boundary conditionds.
    '''

    Nx, Ny = x_in.shape
    x_pad = np.zeros((Nx + 2, Ny + 2))
    x_pad[1:Nx+1, 1:Ny+1] = x_in[:, :]

    x_pad[0,    :] = x_pad[1,  :]
    x_pad[Nx+1, :] = x_pad[Nx, :]
    x_pad[:,    0] = x_pad[:,  1]
    x_pad[:, Ny+1] = x_pad[:, Ny]

    Ax = np.zeros((Nx, Ny))
    for ix, iy in np.ndindex(Ax.shape):
        Ax[ix, iy] = laplace_2d(x_pad, ix+1, iy+1)

    return Ax



def mk_Heat_2d(Nx, Ny, bc="dirichlet", xlo=0, xhi=0, ylo=0, yhi=0,dx=1,dt=1,nu=1):
    '''
    mk_Heat_2d(Nx, Ny, bc="dirichlet", xlo=0, xhi=0, ylo=0, yhi=0,dx=1,dt=1,nu=1)

    Generates discretized Heat Equation (BTCS- backward in time and centered in space) LHS operator as a stencil operation for a N-cell 2D grid, for
    given boundary conditions.

    We assume the thermal conductivity (\nu) is isotropic and constant.
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


    def build_bc_3(x_in):
        x_out = np.zeros((Nx + 2, Ny + 2))

        x_out[1:Nx+1, 1:Ny+1] = x_in[:, :]

        x_out[:,    0] = -30*(x_out[:,    0]-1)*(x_out[:,    0]+1)
        x_out[:, Ny+1] = 2*x_out[:, Ny] -x_out[:, Ny-1]
        x_out[0,    :] = xlo
        x_out[Nx+1, :] = xhi

        return x_out

    Lambda=nu*dt/dx**2.0 # Discretization factor
    def Heat_2d(x, i, j):
        return ((1+4*Lambda)*x[i, j] - Lambda*x[i-1, j] - Lambda*x[i+1, j] - Lambda*x[i, j-1] - Lambda*x[i, j+1])


    if bc == "dirichlet":
        op  = lambda x, i, j: Heat_2d(build_bc_1(x), i, j)
    elif bc == "periodic":
        op  = lambda x, i, j: Heat_2d(build_bc_2(x), i, j)
    elif bc == "channel":
        op  = lambda x, i, j: Heat_2d(build_bc_3(x), i, j)
    else:
        raise RuntimeError(f"bc={bc} not implemented")

    return lambda x: appl_2d(op, x, Nx, Ny)



def mk_Advect_2d_RHS(Nx, Ny,Vx,Vy, bc="dirichlet", xlo=0, xhi=0, ylo=0, yhi=0,dx=1,dt=1):
    '''
    mk_Advect_2d_RHS(Nx, Ny,Vx,Vy, bc="dirichlet", xlo=0, xhi=0, ylo=0, yhi=0,dx=1,dt=1)

    Generates discretized (upwind) 2D Advection Equation  RHS operator as a stencil operation for a N-cell 2D grid, for
    given boundary conditions.

    V can be variable, but note that this operator is applied on the RHS(its an explicit method).
    So, care must be taken to ensure the CFL condition is satisfied
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

    def build_bc_3(x_in):
        x_out = np.zeros((Nx + 2, Ny + 2))

        x_out[1:Nx+1, 1:Ny+1] = x_in[:, :]

        x_out[:,    0] = -30*(x_out[:,    0]-1)*(x_out[:,    0]+1)
        x_out[:, Ny+1] = 2*x_out[:, Ny] -x_out[:, Ny-1]
        x_out[0,    :] = xlo
        x_out[Nx+1, :] = xhi

        return x_out


    def Advect_2d_RHS(x, i, j):
        if (Vx[i,j]>0):
            d_dx= (x[i,j]-x[i,j-1])/(dx)
        elif (Vx[i,j]<0):
            d_dx= (x[i,j+1]-x[i,j])/(dx)
        else: d_dx=0

        if (Vy[i,j]>0):
            d_dy= (x[i,j]-x[i-1,j])/(dx)
        elif (Vy[i,j]<0):
            d_dy= (x[i+1,j]-x[i,j])/(dx)
        else: d_dy=0

        Advect = -(Vx[i,j]*d_dx+Vy[i,j]*d_dy)
        return (x[i,j]+dt*Advect)


    if bc == "dirichlet":
        op  = lambda x, i, j: Advect_2d_RHS(build_bc_1(x), i, j)
    elif bc == "periodic":
        op  = lambda x, i, j: Advect_2d_RHS(build_bc_2(x), i, j)
    elif bc == "channel":
        op  = lambda x, i, j: Advect_2d_RHS(build_bc_3(x), i, j)
    else:
        raise RuntimeError(f"bc={bc} not implemented")

    return lambda x: appl_2d(op, x, Nx, Ny)




##_____________________________________________________________________________
# Tensor Linop code.
# Current implementation leads to slow run time relative to other backprop computations


def appl_2d_Tensor(op, x, Nx, Ny):
    Num=x.size(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Ax_Tensor = torch.zeros(Num,Nx, Ny,requires_grad=False).to(device)
    for ix, iy in np.ndindex(Ax_Tensor[0].shape):
        Ax_Tensor[:,ix, iy] = op(x, ix+1, iy+1)

    return Ax_Tensor



def mk_laplace_2d_Tensor(Nx, Ny,bc="dirichlet", xlo=0, xhi=0, ylo=0, yhi=0):
    '''
    mk_laplace_2d(N, bc="dirichlet", xlo=0, xhi=0, ylo=0, yhi=0)

    Generates laplace operator as a stencil operation for a N-cell 2D grid, for
    given boundary conditionds.
    '''
    def build_bc_1_Tensor(x_in):
        Num=x_in.size(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_out = torch.zeros(Num,Nx + 2, Ny + 2,requires_grad=False).to(device)

        x_out[:,1:Nx+1, 1:Ny+1] = x_in[:,:, :]

        x_out[:,0,    :] = xlo
        x_out[:,Nx+1, :] = xhi
        x_out[:,:,    0] = ylo
        x_out[:,:, Ny+1] = yhi

        return x_out


    def build_bc_2_Tensor(x_in):
        Num=x_in.size(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_out = torch.zeros(Num,Nx + 2, Ny + 2,requires_grad=False).to(device)

        x_out[:,1:Nx+1, 1:Ny+1] = x_in[:,:, :]

        x_out[:,0,    1:Nx+1] = x_in[:,-1, :]
        x_out[:,Nx+1, 1:Nx+1] = x_in[:,0,  :]
        x_out[:,1:Nx+1,    0] = x_in[:,:, -1]
        x_out[:,1:Nx+1, Ny+1] = x_in[:,:,  0]

        return x_out


    def laplace_2d_Tensor(x, i, j):
        return (-4*x[:,i, j] + x[:,i-1, j] + x[:,i+1, j] + x[:,i, j-1] + x[:,i, j+1])


    if bc == "dirichlet":
        op  = lambda x, i, j: laplace_2d_Tensor(build_bc_1_Tensor(x), i, j)
    elif bc == "periodic":
        op  = lambda x, i, j: laplace_2d_Tensor(build_bc_2_Tensor(x), i, j)
    else:
        raise RuntimeError(f"bc={bc} not implemented")

    return lambda x: appl_2d_Tensor(op, x, Nx, Ny)
