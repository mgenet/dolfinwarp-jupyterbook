#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin

import dolfin_cm as dcm

################################################################################

def compute_error(
        val,
        ref):

    if (dolfin.near(ref, 0., eps=1e-9)):
        if (dolfin.near(val, 0., eps=1e-9)):
            return 0.
        else:
            return 1.
    else:
        return val/ref
