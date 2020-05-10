#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Based on this question (and responses) from stackoverflow:
# https://stackoverflow.com/questions/37962271/whats-wrong-with-my-gmres-implementation


import numpy as np
import scipy as sp

from scipy.linalg         import get_blas_funcs, get_lapack_funcs
from scipy.sparse.sputils import upcast


# define consistent linear math that stick with `np.array` (rather than
# `np.matrix`) => this will mean that we're sticking with the "minimal" data
# type for vector data. NOTE: this might cause a performance hit due to
# changing data types.
mat_to_a = lambda a    : np.squeeze(np.asarray(a))
matmul_a = lambda a, b : mat_to_a(np.dot(a, b))


# mathematical indices for python
cidx   = lambda i: i-1  # c-style index from math-style index
midx   = lambda i: i+1  # math-style index from c-style index
mrange = lambda n: range(1, n + 1)


def GMRES(A, b, x0, e, nmax_iter, restart=None, debug=False):
    """
    Quick and dirty GMRES -- TODO: optimize going to larger
    systems.
    """

    # TODO: you can use this to make the problem agnostic to complex numbers
    # # Defining xtype as dtype of the problem, to decide which BLAS functions
    # # import.
    # xtype = upcast(x0.dtype, b.dtype)

    # Defining dimension
    dimen = len(x0)

    # TODO: use BLAS functions
    # # Get fast access to underlying BLAS routines
    # [lartg] = get_lapack_funcs(['lartg'], [x0] )
    # if np.iscomplexobj(np.zeros((1,), dtype=xtype)):
    #     [axpy, dotu, dotc, scal] =\
    #         get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal'], [x0])
    # else:
    #     # real type
    #     [axpy, dotu, dotc, scal] =\
    #         get_blas_funcs(['axpy', 'dot', 'dot', 'scal'], [xO])

    # TODOs for this function:
    # 1. list -> numpy.array <= better memory access
    # 2. don't append to lists -> prealoc and slice
    # 3. add documentation -- this will probably never happen :P
    
    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0

    # TODO: is the old code (below) faster?
    # r = b - np.asarray(np.dot(A, x0)).reshape(-1)
    r = b - matmul_a(A, x0)

    # Set number of outer loops based on the value of `restart`
    n_outer = 1
    if restart is not None:
        n_outer = int(restart)

    x     = [x0]
    x_sol = x0

    for l in mrange(n_outer):
        q    = [None] * (nmax_iter)
        q[cidx(1)] = r / np.linalg.norm(r)

        h = np.zeros((nmax_iter + 1, nmax_iter))

        for k in mrange(min(nmax_iter, dimen)):
            # TODO: is the old code (below) faster?
            # y = np.asarray(np.dot(A, q[k])).reshape(-1)
            y = matmul_a(A, q[cidx(k)])

            # Modified Grahm-Schmidt
            for j in range(1, k+1):
                h[cidx(j), cidx(k)] = np.dot(q[cidx(j)], y)
                y = y - h[cidx(j), cidx(k)] * q[cidx(j)]

            h[cidx(k + 1), cidx(k)] = np.linalg.norm(y)

            if (h[cidx(k + 1), cidx(k)] != 0 and k != nmax_iter):
                q[cidx(k + 1)] = y / h[cidx(k + 1), cidx(k)]

            if debug:
                beta    = np.zeros(nmax_iter + 1)
                beta[0] = np.linalg.norm(r)
                y       = np.linalg.lstsq(h, beta, rcond=None)[0]
                g       = np.dot(np.asarray(q[:cidx(k)]).transpose(), y[:cidx(k)])
                x.append(x_sol + g)


        beta    = np.zeros(nmax_iter + 1)
        beta[0] = np.linalg.norm(r)
        y       = np.linalg.lstsq(h, beta, rcond=None)[0]
        g       = np.dot(np.asarray(q).transpose(), y)
        
        x_sol   = x_sol + g
        x.append(x_sol)
 
        r = b - matmul_a(A, x_sol)

        # Break out if the residual is lower than threshold
        if np.linalg.norm(r)/normb < e*A.shape[0]:
            break

 
    return x


def apply_givens(Q, v, k):
    """
    Apply the first k Givens rotations in Q to the vector v.
    Arguments
    ---------
        Q: list, list of consecutive 2x2 Givens rotations
        v: array, vector to apply the rotations to
        k: int, number of rotations to apply
    Returns
    -------
        v: array, that is changed in place.
    """

    for j in range(k):
        Qloc = Q[j]
        # TODO: why sp.dot and not np.dot?
        v[j:j+2] = sp.dot(Qloc, v[j:j+2])



def GMRES_R(A, b, x0, tol, max_outer, max_inner, restart=None):
    """
    Quick and dirty GMRES -- TODO: optimize mem footprint when going to larger
    systems.
    """

    X = x0

    # Defining xtype as dtype of the problem, to decide which BLAS functions
    # import.
    xtype = upcast(X.dtype, b.dtype)

    # Get fast access to underlying BLAS routines
    # dotc is the conjugate dot, dotu does no conjugation

    [lartg] = get_lapack_funcs(['lartg'], [X] )
    if np.iscomplexobj(np.zeros((1,), dtype=xtype)):
        [axpy, dotu, dotc, scal] =\
            get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal'], [X])
    else:
        # real type
        [axpy, dotu, dotc, scal] =\
            get_blas_funcs(['axpy', 'dot', 'dot', 'scal'], [X])

    # Make full use of direct access to BLAS by defining own norm
    def norm(z):
        return np.sqrt(np.real(dotc(z, z)))

    # Defining dimension
    dimen = len(X)


    # TODOs for this function:
    # 1. list -> numpy.array <= better memory access
    # 2. don't append to lists -> prealoc and slice
    # 3. lapack replacemnt for matmul_a?
    # 4. clean up this function!
    # 3. add documentation -- this will probably never happen :P

    r = b - matmul_a(A, x0)

    normr = norm(r)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0

    iteration = 0
    x = list()

    # Here start the GMRES
    for outer in range(max_outer):
        # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
        # Space required is O(dimen*max_inner).
        # NOTE:  We are dealing with row-major matrices, so we traverse in a
        #        row-major fashion,
        #        i.e., H and V's transpose is what we store.

        Q = []  # Initialzing Givens Rotations
        # Upper Hessenberg matrix, which is then
        # converted to upper triagonal with Givens Rotations

        H = np.zeros((max_inner + 1, max_inner + 1), dtype=xtype)
        V = np.zeros((max_inner + 1, dimen), dtype=xtype)  # Krylov space

        # vs store the pointers to each column of V.
        # This saves a considerable amount of time.
        vs = []

        # v = r/normr
        V[0, :] = scal(1.0/normr, r)  # scal wrapper of dscal --> x = a*x
        vs.append(V[0, :])

        # Saving initial residual to be used to calculate the rel_resid
        if iteration == 0:
            res_0 = normb
 
        # RHS vector in the Krylov space
        g    = np.zeros((dimen, ), dtype=xtype)
        g[0] = normr

        for inner in range(max_inner):
            # New search direction
            v    = V[inner+1, :]  # pointer!
            v[:] = matmul_a(A, vs[-1])
            vs.append(v)

            # Modified Gram Schmidt
            for k in range(inner+1):
                vk          = vs[k]
                alpha       = dotc(vk, v)
                H[inner, k] = alpha
                v[:]        = axpy(vk, v, dimen, -alpha)  # y := a*x + y
                #axpy is a wrapper for daxpy (blas function)

            normv = norm(v)
            H[inner, inner + 1] = normv

            # Check for breakdown
            if H[inner, inner + 1] != 0.0:
                v[:] = scal(1.0/H[inner, inner + 1], v)

            # Apply for Givens rotations to H
            if inner > 0:
                apply_givens(Q, H[inner, :], inner)

            # Calculate and apply next complex-valued Givens rotations

            # If max_inner = dimen, we don't need to calculate, this
            # is unnecessary for the last inner iteration when inner = dimen -1

            if inner != dimen - 1:
                if H[inner, inner + 1] != 0:
                    # lartg is a lapack function that computes the parameters
                    # for a Givens rotation
                    [c, s, _] = lartg(H[inner, inner], H[inner, inner + 1])
                    Qblock = np.array([[c, s], [-np.conjugate(s),c]], dtype=xtype)
                    Q.append(Qblock)

                    #Apply Givens Rotations to RHS for the linear system in
                    # the krylov space. TODO: why sp.dot and not np.dot?
                    g[inner:inner+2] = sp.dot(Qblock, g[inner:inner+2])

                    #Apply Givens rotations to H
                    H[inner, inner] = dotu(Qblock[0,:], H[inner, inner:inner+2])
                    H[inner, inner+1] = 0.0

            iteration+= 1

            if inner < max_inner-1:
                normr = abs(g[inner+1])
                rel_resid = normr/res_0

                if rel_resid < tol:
                    break

        # end inner loop, back to outer loop

        # Find best update to X in Krylov Space V.  Solve inner X inner system.
        y      = sp.linalg.solve(H[0:inner+1, 0:inner+1].T, g[0:inner+1])
        update = np.ravel(sp.mat(V[:inner+1, :]).T.dot(y.reshape(-1,1)))
        X      = X + update
        aux    = matmul_a(A, X)
        r      = b - aux
        
        normr = norm(r)
        rel_resid = normr/res_0

        x.append(X)

        # test for convergence
        if rel_resid < tol:
            print('GMRES solve')
            print(f'Converged after {iteration} iterations to a residual of {rel_resid}')
            return x

    #end outer loop

    return x