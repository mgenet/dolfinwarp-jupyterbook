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

class SurfaceTension0LoadingOperator(Operator):

    def __init__(self,
            u,
            u_test,
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

        dim = u.ufl_shape[0]
        I = dolfin.Identity(dim)
        Pi = gamma * (1 + dolfin.inner(
            kinematics.epsilon,
            I - dolfin.outer(N,N))) * self.measure
        self.res_form = dolfin.derivative(Pi, u, u_test) # MG20211220: Is that correct?!



    def set_value_at_t_step(self,
            t_step):

        self.tv_gamma.set_value_at_t_step(t_step)
