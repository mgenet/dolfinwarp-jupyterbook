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
import numpy

import dolfin_cm as dcm

################################################################################

class SubSol():



    def __init__(self,
            name,
            fe,
            init_val=None):

        self.name = name
        self.fe = fe

        if (init_val is None):
            # print("value_shape = "+str(fe.value_shape()))
            self.init_val = numpy.zeros(fe.value_shape())
        else:
            # print("size = "+str(numpy.size(init)))
            self.init_val = init_val
        # print("init_val = "+str(self.init_val))
