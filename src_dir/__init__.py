#!/usr/bin/env python

from .util                 import resid, cidx, midx, mrange, matmul_a, mat_to_a,\
                                  timer, Gauss_pdf,Gauss_pdf_2D ,moving_average,\
                                  OverwriteLast, StatusPrinter
from .linop                import laplace_1d, mk_laplace_1d, mk_laplace_2d,\
                                  mk_laplace_2d_Tensor
from .gmres                import GMRES, GMRES_R
from .benchmarking         import *
from .cnn_predictorOnline2D  import cnn_preconditionerOnline_timed_2D,\
                                    CNNPredictorOnline_2D, PreconditionerTrainer
# from .cnn_collectionOnline2D import CnnOnline_2D
from .cnn_collectionOnline2D import FluidNet2D10, FluidNet2D20, FluidNet2D30
