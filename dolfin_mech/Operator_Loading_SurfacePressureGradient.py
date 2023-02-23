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
            x0,
            x0_test,
            kinematics,
            U,
            U_test,
            Phis0,
            V0, 
            N,
            lbda,
            lbda_test,
            mu,
            mu_test,
            dS,
            dV,
            rho_solid,
            P0_val=None, P0_ini=None, P0_fin=None,
            f_val=None, f_ini=None, f_fin=None):


        self.measure = dV
        self.dS = dS

        self.tv_F = dmech.TimeVaryingConstant(
            val=f_val, val_ini=f_ini, val_fin=f_fin)
        F = self.tv_F.val

        self.tv_P0 = dmech.TimeVaryingConstant(
            val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
        P0 = self.tv_P0.val

        nf = dolfin.dot(N, dolfin.inv(kinematics.F))
        nf_norm = (1/2 * dolfin.inner(nf,nf))**(1/2)
        n = nf/nf_norm

        self.V0 = dolfin.Constant(V0)
        rho0 = dolfin.assemble(Phis0 * self.measure)
        
        x = X + U
        x_tilde = x-x0
        P = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x_tilde, n))

        self.res_form = - dolfin.inner(rho_solid * Phis0 * F, U_test) * self.measure
        self.res_form -= dolfin.inner(-P * n, U_test) * kinematics.J * nf_norm * self.dS
        self.res_form += dolfin.inner((Phis0 * x / dolfin.Constant(rho0) - x0/self.V0), x0_test) * self.measure
        self.res_form += dolfin.inner(-P * n, lbda_test) * kinematics.J * nf_norm * self.dS
        self.res_form +=  dolfin.inner(rho_solid * Phis0 * F , lbda_test) * self.measure
        self.res_form += dolfin.inner(dolfin.cross(x_tilde, -P * n), mu_test) * kinematics.J * nf_norm * self.dS
         

    def set_value_at_t_step(self,
            t_step):       
        self.tv_F.set_value_at_t_step(t_step)
        self.tv_P0.set_value_at_t_step(t_step)
        pass


################################################################################

class SurfacePressureGradient0LoadingOperator(Operator):

    def __init__(self,
            x,
            x0,
            n,
            u_test,
            lbda,
            lbda_test,
            mu,
            mu_test,
            dS,
            dV,
            rho_solid,
            phis,
            P0_val=None, P0_ini=None, P0_fin=None,
            f_val=None, f_ini=None, f_fin=None):


        self.dS = dS
        self.measure = dV

        self.tv_f = dmech.TimeVaryingConstant(
            val=f_val, val_ini=f_ini, val_fin=f_fin)
        f = self.tv_f.val

        self.tv_P0 = dmech.TimeVaryingConstant(
            val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
        P0 = self.tv_P0.val

        x_tilde = x-dolfin.Constant(x0)
        P = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x_tilde, n))

        self.res_form = - dolfin.inner(rho_solid*phis*f, u_test) * self.measure
        self.res_form -= dolfin.inner(-P * n, u_test) * self.dS
        self.res_form += dolfin.inner(dolfin.cross(x_tilde, -P * n), mu_test) * self.dS
        self.res_form += dolfin.inner(rho_solid*phis*f, lbda_test) * self.measure
        self.res_form += dolfin.inner(-P * n, lbda_test) * self.dS


    def set_value_at_t_step(self,
            t_step):
        self.tv_f.set_value_at_t_step(t_step)
        self.tv_P0.set_value_at_t_step(t_step)
        
