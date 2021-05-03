#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2020                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin
import numpy

import dolfin_mech as dmech
from .Problem import Problem

################################################################################

class LinearizedKinematics():



    def __init__(self,
            dim,
            U,
            U_old):

        self.I = dolfin.Identity(dim)

        self.epsilon     = dolfin.sym(dolfin.grad(U    ))
        self.epsilon_old = dolfin.sym(dolfin.grad(U_old))

        self.epsilon_sph     = dolfin.tr(self.epsilon    )/dim * self.I
        self.epsilon_sph_old = dolfin.tr(self.epsilon_old)/dim * self.I

        self.epsilon_dev     = self.epsilon     - self.epsilon_sph
        self.epsilon_dev_old = self.epsilon_old - self.epsilon_sph_old

        self.epsilon_mid = (self.epsilon_old + self.epsilon)/2
