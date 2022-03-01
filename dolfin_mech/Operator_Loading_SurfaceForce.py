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

class SurfaceForceLoadingOperator(Operator):

    def __init__(self,
            U_test,
            kinematics,
            N,
            measure,
            F=None,
            F_ini=None,
            F_fin=None):

        self.measure = measure

        self.tv_F = dmech.TimeVaryingConstant(
            val=F, val_ini=F_ini, val_fin=F_fin)
        F = self.tv_F.val

        FmTN = dolfin.dot(dolfin.inv(kinematics.F).T, N)
        T = dolfin.sqrt(dolfin.inner(FmTN, FmTN)) * F
        self.res_form = - dolfin.inner(T, U_test) * kinematics.J * self.measure



    def set_value_at_t_step(self,
            t_step):

        self.tv_F.set_value_at_t_step(t_step)