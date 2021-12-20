#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech

################################################################################

class Kinematics():



    def __init__(self,
            dim,
            U,
            U_old,
            Q_expr=None,
            w_growth=False,
            w_relaxation=False):

        self.I = dolfin.Identity(dim)

        self.F     = self.I + dolfin.grad(U)
        self.F_old = self.I + dolfin.grad(U_old)

        self.J     = dolfin.det(self.F    )
        self.J_old = dolfin.det(self.F_old)

        self.C     = self.F.T     * self.F
        self.C_old = self.F_old.T * self.F_old

        self.E     = (self.C     - self.I)/2
        self.E_old = (self.C_old - self.I)/2

        if (Q_expr is not None):
            self.E_loc = Q_expr * self.E * Q_expr.T # MG20211215: This should work, right?
            # self.E_loc = dolfin.dot(dolfin.dot(Q_expr, self.E), Q_expr.T)

        self.F_mid = (self.F_old + self.F)/2
        self.C_mid = (self.C_old + self.C)/2
        self.E_mid = (self.E_old + self.E)/2
