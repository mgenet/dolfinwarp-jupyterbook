#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018                                            ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

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
