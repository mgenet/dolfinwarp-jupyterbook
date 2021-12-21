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

class SurfaceTensionLoadingOperator(Operator):

    def __init__(self,
            U,
            U_test,
            kinematics,
            N,
            measure,
            gamma=None,
            gamma_ini=None,
            gamma_fin=None):

        self.measure = measure

        self.tv_gamma = dmech.TimeVaryingConstant(
            val=gamma, val_ini=gamma_ini, val_fin=gamma_fin)
        gamma = self.tv_gamma.val

        FmTN = dolfin.dot(dolfin.inv(kinematics.F).T, N)
        T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
        Pi = gamma * T * kinematics.J * self.measure
        self.res_form = dolfin.derivative(Pi, U, U_test)



    def set_value_at_t_step(self,
            t_step):

        self.tv_gamma.set_value_at_t_step(t_step)
