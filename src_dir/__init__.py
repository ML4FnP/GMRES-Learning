#!/usr/bin/env python

from .util          import resid, cidx, midx, mrange, matmul_a, mat_to_a,\
                           timer, Gauss_pdf, moving_average
from .linop         import laplace_1d, mk_laplace_1d
from .gmres         import GMRES, GMRES_R
from .nn_collection import TwoLayerNet
from .nn_predictor  import nn_preconditioner,nn_preconditioner_timed
from .cnn_predictorOnline  import cnn_preconditionerOnline_timed
from .cnn_collectionOnline import CnnOnline
