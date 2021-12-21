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

class InertiaOperator(Operator):

    def __init__(self,
            U,
            U_test,
            measure,
            rho=None,
            rho_ini=None,
            rho_fin=None):

        self.measure = measure

        self.tv_rho = dmech.TimeVaryingConstant(
            val=rho, val_ini=rho_ini, val_fin=rho_fin)
        rho = self.tv_rho.val

        self.tv_dt = dmech.TimeVaryingConstant(0.)
        dt = self.tv_dt.val

        Pi = (rho/2/dt) * dolfin.inner(U, U)**2 * self.measure
        self.res_form = dolfin.derivative(Pi, U, U_test)



    def set_value_at_t_step(self,
            t_step):

        self.tv_rho.set_value_at_t_step(t_step)



    def set_dt(self,
            dt):

        self.tv_dt.set_value(dt)
