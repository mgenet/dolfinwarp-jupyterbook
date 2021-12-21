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

class InverseKinematics():



    def __init__(self,
            dim,
            U,
            U_old):

        self.I = dolfin.Identity(dim)

        self.f     = self.I + dolfin.grad(U    )
        self.f_old = self.I + dolfin.grad(U_old)

        self.F     = dolfin.inv(self.f    )
        self.F_old = dolfin.inv(self.f_old)

        self.J     = dolfin.det(self.F    )
        self.J_old = dolfin.det(self.F_old)

        self.C     = self.F.T     * self.F
        self.C_old = self.F_old.T * self.F_old

        self.C_inv = dolfin.inv(self.C)

        self.IC  = dolfin.tr(self.C)
        self.IIC = (dolfin.tr(self.C)*dolfin.tr(self.C) - dolfin.tr(self.C*self.C))/2

        self.E     = (self.C     - self.I)/2
        self.E_old = (self.C_old - self.I)/2

        self.F_bar   = self.J**(-1./3) * self.F
        self.C_bar   = self.F_bar.T * self.F_bar
        self.IC_bar  = dolfin.tr(self.C_bar)
        self.IIC_bar = (dolfin.tr(self.C_bar)*dolfin.tr(self.C_bar) - dolfin.tr(self.C_bar*self.C_bar))/2
        self.E_bar   = (self.C_bar - self.I)/2
