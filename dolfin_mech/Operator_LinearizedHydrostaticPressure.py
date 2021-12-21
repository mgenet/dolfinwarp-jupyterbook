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

class LinearizedHydrostaticPressureOperator(Operator):

    def __init__(self,
            u,
            u_test,
            kinematics,
            p,
            measure):

        epsilon_test = dolfin.derivative(
            kinematics.epsilon, u, u_test)

        self.p = p
        self.measure = measure
        self.res_form = -self.p * dolfin.tr(epsilon_test) * self.measure
