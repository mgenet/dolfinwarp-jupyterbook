#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin

import dolfin_cm as dcm
from .Material_Elastic import ElasticMaterial

################################################################################

class TawhaiElasticMaterial(ElasticMaterial):



    def __init__(self):

            self.alpha     = dolfin.Constant(0.433)
            self.beta      = dolfin.Constant(-0.611)
            self.gamma     = dolfin.Constant(2.5)



    def get_free_energy(self,
            U=None,
            C=None):

        if (C is None):
            dim = U.ufl_shape[0]
            I = dolfin.Identity(dim)
            F = I + dolfin.grad(U)
            C = F.T * F

        # dim   = U.ufl_shape[0]
        I     = dolfin.Identity(3)
        JF    = dolfin.sqrt(dolfin.det(C))
        I1    = dolfin.tr(C)
        I2    = (1./2.)*((dolfin.tr(C))**2 - dolfin.tr(C * C))
        C_inv = dolfin.inv(C)

        expo    = dolfin.exp(self.alpha * I1 * JF**(-2./3.) + self.beta * I2 * JF**(-4./3.))
        Psi     = self.gamma * expo

        dpsidI1 = self.gamma * self.alpha * JF**(-2./3.) * expo
        dpsidI2 = self.gamma * self.beta * JF**(-4./3.) * expo
        dpsidJ  = self.gamma * (self.alpha * I1 * (-2./3.) * JF**(-5./3.) + self.beta * I2 * (-4./3.) * JF**(-7./3.)) * expo
        Sigma   = 2 * (dpsidI1 * I + dpsidI2 * (I1 * I - C) + dpsidJ * JF / 2 * C_inv)

        return Psi, Sigma
