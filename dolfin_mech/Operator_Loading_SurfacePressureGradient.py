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

import numpy




class x0DirectOperator(Operator):

    def __init__(self,
            x0,
            x0_test,
            X,
            U,
            phis0,
            V0, 
            rho0,
            measure):

        self.V0 = dolfin.Constant(V0)
        self.measure = measure

        self.res_form =  dolfin.inner(( (X+U)*phis0/rho0 - x0/self.V0), x0_test) * self.measure

        
################################################################################


class CoefficientLambda(Operator):
    def _init(self,
    X,
    x0,
    U,
    N,
    Phis0,
    kinematics,
    lbda,
    lbda_test,
    mu,
    dS,
    dV)

        self.dV = dV
        self.dS = dS

        x = X + U

        x_tilde = x-x0

        n = dolfin.dot(N,dolfin.inv(kinematics.F))

        p = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x_tilde,n))

        self.res_form = dolfin.dot(p*n, lbda_test) * kinematics.J * self.dS + dolfin.dot(dolfin.Constant(9.81e-3)*Phis0, lbda_test) * self.dV


################################################################################

class CoefficientMu(Operator):
    def _init(self,
    X,
    x0,
    U,
    N,
    Phis0,
    kinematics,
    lbda,
    mu,
    mu_test,
    dS)



        self.dS = dS

        x = X + U

        x_tilde = x-x0

        n = dolfin.dot(N,dolfin.inv(kinematics.F))

        p = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x_tilde,n))

        self.res_form = dolfin.dot(dolfin.cross(x_tilde, n), mu_test) * kinematics.J * self.dS 




################################################################################


class SurfacePressureGradientLoadingOperator(Operator):

    def __init__(self,
            X,
            x0,
            U,
            U_test,
            kinematics,
            N,
            M1_d,
            M2_d,
            M3_d,
            M4_d,
            second_membre,
            measure,
            X0_val=None, X0_ini=None, X0_fin=None,
            P0_val=None, P0_ini=None, P0_fin=None):

        


        self.measure = measure

        self.tv_P0 = dmech.TimeVaryingConstant(
            val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
        P0 = self.tv_P0.val

        # self.tv_x0 = dmech.TimeVaryingConstant(
        #     val=None, val_ini=[0,0,0], val_fin=x0)
        # x0 = self.tv_x0.val

        # print("M1_d", M1_d)
        # print(type(M1_d))

        I = dolfin.as_matrix(numpy.eye(3))

        A = dolfin.inv(M1_d-M2_d*dolfin.inv(M4_d)*M3_d)
        B = -A*M2_d*dolfin.inv(M4_d)
        C = -dolfin.inv(M4_d)*M3_d*A
        D = dolfin.inv(M4_d)*(I-M3_d*B)

        F = dolfin.as_vector([second_membre[0], second_membre[1], second_membre[2]])
        G = dolfin.as_vector([second_membre[3], second_membre[4], second_membre[5]])

        lbda = A*F + B*G

        mu = C*F + D*G

        # print("lbda=", lbda)
        # print("type lbda", type(lbda))

        # mu = dolfin.inv(B)*(F-A*lbda)

        # print("mu, lbda computed")


        # self.tv_lbda = dmech.TimeVaryingConstant(
        #     val=None, val_ini=[0,0,0], val_fin=[lbda[0], lbda[1], lbda[2]])
        # lbda = self.tv_lbda.val

        # self.tv_mu = dmech.TimeVaryingConstant(
        #     val=None, val_ini=[0,0,0], val_fin=mu)
        # mu = self.tv_mu.val

    

        x = X + U

        n = dolfin.dot(N,dolfin.inv(kinematics.F))

        # print("trying")
        # print("lbda=", lbda)
        # print("type lbda", type(lbda))
        # print("len(lbda)", len(lbda))
        # print("converting lbda", dolfin.as_vector(lbda))


        P = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x-x0, n)) 

        
       
        T = dolfin.dot(-P * N, dolfin.inv(kinematics.F))
        self.res_form = - dolfin.inner(T, U_test) * kinematics.J * self.measure



    def set_value_at_t_step(self,
            t_step):

        # self.tv_x0.set_value_at_t_step(t_step)
        
        self.tv_P0.set_value_at_t_step(t_step)
        # self.tv_lbda.set_value_at_t_step(t_step)
        # self.tv_mu.set_value_at_t_step(t_step)

################################################################################

class SurfacePressureGradient0LoadingOperator(Operator):

    def __init__(self,
            X,
            X0,
            U_test,
            N,
            vect,
            measure,
            X0_val=None, X0_ini=None, X0_fin=None,
            P0_val=None, P0_ini=None, P0_fin=None):



        self.measure = measure
        
        self.tv_P0 = dmech.TimeVaryingConstant(
            val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
        P0 = self.tv_P0.val

        self.tv_X0 = dmech.TimeVaryingConstant(
            val=X0_val, val_ini=X0_ini, val_fin=X0_fin)
        X0 = self.tv_X0.val

        self.tv_lbda = dmech.TimeVaryingConstant(
            val=None, val_ini=[0,0,0], val_fin=vect[0])
        lbda = self.tv_lbda.val

        self.tv_mu = dmech.TimeVaryingConstant(
            val=None, val_ini=[0,0,0], val_fin=vect[1])
        mu = self.tv_mu.val
        

        P = P0 + dolfin.dot(lbda, N) + dolfin.dot(mu, dolfin.cross(X-X0, N))
           
        self.res_form = - dolfin.inner(-P * N, U_test) * self.measure



    def set_value_at_t_step(self,
            t_step):

        self.tv_X0.set_value_at_t_step(t_step)

        self.tv_P0.set_value_at_t_step(t_step)
        self.tv_lbda.set_value_at_t_step(t_step)
        self.tv_mu.set_value_at_t_step(t_step)
        
