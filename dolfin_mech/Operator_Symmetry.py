#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class SymmetryOperator(Operator):

    def __init__(self,
            tensor,
            tensor_test,
            measure):

        self.measure = measure
        self.res_form = dolfin.inner(tensor.T - tensor, tensor_test) * self.measure
        # self.res_form = (tensor[1,0] - tensor[0,1]) * tensor_test[1,0] * self.measure
