#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Based on this question (and responses) from stackoverflow:
# https://stackoverflow.com/questions/37962271/whats-wrong-with-my-gmres-implementation


import numpy as np



def GMRES(A, b, x0, e, nmax_iter, restart=None):
    """
    Quick and dirty GMRES -- TODO: optimize mem footprint when going to larger
    systems.
    """

    # TODOs for this function:
    # 1. list -> numpy.array <= better memory access
    # 2. don't append to lists -> prealoc and slice
    # 3. add documentation -- this will probably never happen :P
    
    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0

    #TODO: use matmul_a for consistency here?
    r = b - np.asarray(np.dot(A, x0)).reshape(-1)
    
    x    = [r]
    q    = [0] * (nmax_iter)
    q[0] = r / np.linalg.norm(r)

    h = np.zeros((nmax_iter + 1, nmax_iter))

    for k in range(min(nmax_iter, A.shape[0])):
        y = np.asarray(np.dot(A, q[k])).reshape(-1)

        for j in range(k + 1):
            h[j, k] = np.dot(q[j], y)
            y = y - h[j, k] * q[j]
        h[k + 1, k] = np.linalg.norm(y)
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            q[k + 1] = y / h[k + 1, k]

        b_inner    = np.zeros(nmax_iter + 1)
        b_inner[0] = np.linalg.norm(r)

        result  = np.linalg.lstsq(h, b_inner, rcond=None)[0]
        g       = np.dot(np.asarray(q).transpose(), result)
        r_inner = b - (g + x0)

        x.append(g + x0)

        # break out if the residual is lower than threshold
        if np.linalg.norm(r_inner)/normb < e*A.shape[0]:
            break

 
    return x


# define consistent linear math that stick with `np.array` (rather than
# `np.matrix`) => this will mean that we're sticking with the "minimal" data
# type for vector data. NOTE: this might cause a performance hit due to
# changing data types.
mat_to_a = lambda a    : np.squeeze(np.asarray(a))
matmul_a = lambda a, b : mat_to_a(np.dot(a, b))
