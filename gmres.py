#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Based on this question (and responses) from stackoverflow:
# https://stackoverflow.com/questions/37962271/whats-wrong-with-my-gmres-implementation



import numpy as np

def GMRES(A, b, x0, e, nmax_iter, restart=None):
    r = b - np.asarray(np.dot(A, x0)).reshape(-1)

    x = []
    q = [0] * (nmax_iter)

    x.append(r)

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

        b = np.zeros(nmax_iter + 1)
        b[0] = np.linalg.norm(r)

        result = np.linalg.lstsq(h, b, rcond=None)[0]

        x.append(np.dot(np.asarray(q).transpose(), result) + x0)

    return x
