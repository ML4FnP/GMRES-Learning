#!/usr/bin/env python

from .util          import resid, cidx, midx, mrange, matmul_a, mat_to_a 
from .gmres         import GMRES, GMRES_R
from .nn_collection import TwoLayerNet
from .nn_predictor  import nn_preconditioner