#!/usr/bin/env python

from .util          import resid, cidx, midx, mrange, matmul_a, mat_to_a, laplace_1d,timer
from .gmres         import GMRES, GMRES_R
from .nn_collection import TwoLayerNet
from .nn_predictor  import nn_preconditioner,nn_preconditioner_timed