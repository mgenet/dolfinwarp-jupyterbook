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
            gamma_val=None, gamma_ini=None, gamma_fin=None):

        self.measure = measure

        self.tv_gamma = dmech.TimeVaryingConstant(
            val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
        gamma = self.tv_gamma.val

        # self.tv_gamma.surface_change_rate(kinematics, dt)
        print("gamma =" +str(gamma))
        FmTN = dolfin.dot(dolfin.inv(kinematics.F).T, N)
        T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
        Pi = gamma * T * kinematics.J * self.measure
        self.res_form = dolfin.derivative(Pi, U, U_test)

        self.kinematics=kinematics

    # def surface_change_rate(self,
    #         dt):

    #     self.tv_gamma.surface_change_rate(self.kinematics, dt)

    def set_value_at_t_step(self,
            t_step):

        self.tv_gamma.set_value_at_t_step(t_step)
        print("t_step =" +str(t_step))
        print("value at t_step = " +str(self.tv_gamma.set_value_at_t_step(t_step)))

    def returne_surface_rate(self):
        self.tv_gamma.surface_change_rate()

################################################################################

class SurfaceTension0LoadingOperator(Operator):

    def __init__(self,
            u,
            u_test,
            kinematics,
            N,
            measure,
            gamma_val=None, gamma_ini=None, gamma_fin=None):

        self.measure = measure

        self.tv_gamma = dmech.TimeVaryingConstant(
            val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
        gamma = self.tv_gamma.val

        dim = u.ufl_shape[0]
        I = dolfin.Identity(dim)
        Pi = gamma * (1 + dolfin.inner(
            kinematics.E,
            I - dolfin.outer(N,N))) * self.measure
        self.res_form = dolfin.derivative(Pi, u, u_test) # MG20211220: Is that correct?!



    def set_value_at_t_step(self,
            t_step):

        self.tv_gamma.set_value_at_t_step(t_step)
