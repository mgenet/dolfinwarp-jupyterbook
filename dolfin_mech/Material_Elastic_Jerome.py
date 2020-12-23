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

import dolfin_mech as dmech
from .Material_Elastic import ElasticMaterial

################################################################################

class JeromeElasticMaterial(ElasticMaterial):



    def __init__(self,
            parameters):

        self.Kappa1 = dolfin.Constant(parameters["Kappa1"])
        self.Kappa2 = dolfin.Constant(parameters["Kappa2"])
        self.Bulk   = dolfin.Constant(parameters["Bulk"])



    def get_free_energy(self,
            C=None):

        assert (C.ufl_shape[0] == C.ufl_shape[1])
        dim = C.ufl_shape[0]
        I = dolfin.Identity(dim)

        JF     = dolfin.sqrt(dolfin.det(C))
        IC     = dolfin.tr(C)
        ICbar  = IC * JF**(-2./3)
        IIC    = (dolfin.tr(C)*dolfin.tr(C) - dolfin.tr(C*C))/2
        IICbar = IIC * JF**(-4./3)
        C_inv = dolfin.inv(C)

        Psi = self.Kappa1 * (ICbar  - 3)\
            + self.Kappa2 * (IICbar - 3)\
            + self.Bulk   * (JF - 1 - dolfin.ln(JF))

        C = dolfin.variable(C)
        Sigma = dolfin.diff(Psi, C)

        return Psi, Sigma
