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

class LinearizedIncompressibilityOperator(Operator):

    def __init__(self,
            kinematics,
            p_test,
            measure):

        self.measure = measure
        self.res_form = -dolfin.tr(kinematics.epsilon) * p_test * self.measure
