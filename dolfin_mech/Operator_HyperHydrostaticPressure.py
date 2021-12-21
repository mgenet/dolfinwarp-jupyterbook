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

class HyperHydrostaticPressureOperator(Operator):

    def __init__(self,
            U,
            U_test,
            kinematics,
            P,
            measure):

        dJ_test = dolfin.derivative(
            kinematics.J,
            U,
            U_test)

        self.P = P
        self.measure = measure
        self.res_form = - self.P * dJ_test * self.measure
