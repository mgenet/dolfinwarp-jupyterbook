#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech

################################################################################

class SubSol():



    def __init__(self,
            name,
            fe,
            init_val=None,
            init_field=None):

        self.name = name
        self.fe = fe

        assert (init_val is None) or (init_field is None)
        if ((init_val is None) and (init_field is None)):
            # print("value_shape = "+str(fe.value_shape()))
            if (fe is dolfin.FiniteElement):
                self.init_val = 0.
            else:
                self.init_val = numpy.zeros(fe.value_shape())
            self.init_field = None
        elif (init_val is not None):
            # print("size = "+str(numpy.size(init)))
            self.init_val   = init_val
            self.init_field = None
        elif (init_field is not None):
            self.init_val   = None
            self.init_field = init_field
        # print("init_val = "+str(self.init_val))