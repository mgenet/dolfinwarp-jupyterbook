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

class LinearizedKinematics():



    def __init__(self,
            dim,
            U,
            U_old):

        self.dim = dim
        self.I = dolfin.Identity(self.dim)

        self.epsilon     = dolfin.sym(dolfin.grad(U    ))
        self.epsilon_old = dolfin.sym(dolfin.grad(U_old))

        self.epsilon_sph     = dolfin.tr(self.epsilon    )/self.dim * self.I
        self.epsilon_sph_old = dolfin.tr(self.epsilon_old)/self.dim * self.I

        self.epsilon_dev     = self.epsilon     - self.epsilon_sph
        self.epsilon_dev_old = self.epsilon_old - self.epsilon_sph_old

        self.epsilon_mid = (self.epsilon_old + self.epsilon)/2
