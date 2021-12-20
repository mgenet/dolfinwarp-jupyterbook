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

class LinearizedSurfaceTensionOperator(Operator):

    def __init__(self,
            u,
            u_test,
            kinematics,
            N,
            measure,
            beta=None,
            beta_ini=None,
            beta_fin=None):

        self.measure = measure

        self.tv_beta = dmech.TimeVaryingConstant(
            val=beta,
            val_ini=beta_ini,
            val_fin=beta_fin)
        beta = self.tv_beta.val

        dim = u.ufl_shape[0]
        I = dolfin.Identity(dim)
        Pi = beta * (1 + dolfin.inner(
            kinematics.epsilon,
            I - dolfin.outer(N,N))) * self.measure
        self.res_form = dolfin.derivative(Pi, u, u_test)



    def set_value_at_t_step(self,
            t_step):

        self.tv_beta.set_value_at_t_step(t_step)
