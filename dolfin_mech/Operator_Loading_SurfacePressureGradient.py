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

class SurfacePressureGradientLoadingOperator(Operator):

    def __init__(self,
            X,
            U,
            v,
            U_test,
            kinematics,
            N,
            V0,
            measure,
            X0_val=None, X0_ini=None, X0_fin=None,
            N0_val=None, N0_ini=None, N0_fin=None,
            P0_val=None, P0_ini=None, P0_fin=None,
            F0 = None,
            DP_val=None, DP_ini=None, DP_fin=None):

        self.measure = measure

        self.tv_X0 = dmech.TimeVaryingConstant(
            val=X0_val, val_ini=X0_ini, val_fin=X0_fin)
        X0 = self.tv_X0.val
        self.tv_N0 = dmech.TimeVaryingConstant(
            val=N0_val, val_ini=N0_ini, val_fin=N0_fin)
        N0 = self.tv_N0.val
        self.tv_P0 = dmech.TimeVaryingConstant(
            val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
        P0 = self.tv_P0.val
        if (DP_fin is not None):
            self.tv_DP = dmech.TimeVaryingConstant(
                val=DP_val, val_ini=DP_ini, val_fin=DP_fin)
        else :
            self.tv_DP = dmech.TimeVaryingConstant(
                 val=None, val_ini=0., val_fin=F0*V0)
        DP = self.tv_DP.val / v
            # DP = F0*V0 / v
        
        # print(DP_fin)

        x = X + U
        P = P0 + DP * dolfin.inner(x - X0, N0)
        
        T = dolfin.dot(-P * N, dolfin.inv(kinematics.F))
        self.res_form = - dolfin.inner(T, U_test) * kinematics.J * self.measure



    def set_value_at_t_step(self,
            t_step):

        self.tv_X0.set_value_at_t_step(t_step)
        self.tv_N0.set_value_at_t_step(t_step)
        self.tv_P0.set_value_at_t_step(t_step)
        if (hasattr(self, "tv_DP")):
            self.tv_DP.set_value_at_t_step(t_step)

################################################################################

class SurfacePressureGradient0LoadingOperator(Operator):

    def __init__(self,
            X,
            U_test,
            N,
            measure,
            X0_val=None, X0_ini=None, X0_fin=None,
            N0_val=None, N0_ini=None, N0_fin=None,
            P0_val=None, P0_ini=None, P0_fin=None,
            DP_val=None, DP_ini=None, DP_fin=None):

        self.measure = measure

        self.tv_X0 = dmech.TimeVaryingConstant(
            val=X0_val, val_ini=X0_ini, val_fin=X0_fin)
        X0 = self.tv_X0.val
        self.tv_N0 = dmech.TimeVaryingConstant(
            val=N0_val, val_ini=N0_ini, val_fin=N0_fin)
        N0 = self.tv_N0.val
        self.tv_P0 = dmech.TimeVaryingConstant(
            val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
        P0 = self.tv_P0.val
        self.tv_DP = dmech.TimeVaryingConstant(
            val=DP_val, val_ini=DP_ini, val_fin=DP_fin)
        DP = self.tv_DP.val

        P = P0 + DP * dolfin.inner(X - X0, N0)

        self.res_form = - dolfin.inner(-P * N, U_test) * self.measure



    def set_value_at_t_step(self,
            t_step):

        self.tv_X0.set_value_at_t_step(t_step)
        self.tv_N0.set_value_at_t_step(t_step)
        self.tv_P0.set_value_at_t_step(t_step)
        self.tv_DP.set_value_at_t_step(t_step)
