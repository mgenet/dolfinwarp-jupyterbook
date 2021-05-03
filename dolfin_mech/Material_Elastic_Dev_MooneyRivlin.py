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
from .Material_Elastic_Dev import DevElasticMaterial

################################################################################

class MooneyRivlinDevElasticMaterial(DevElasticMaterial):



    def __init__(self,
            parameters):

        if ("C1" in parameters):
            self.C1 = dolfin.Constant(parameters["C1"])
        if ("C2" in parameters):
            self.C2 = dolfin.Constant(parameters["C2"])



    def get_free_energy(self,
            U=None,
            C=None):

        if (C is None):
            dim = U.ufl_shape[0]
            I = dolfin.Identity(dim)
            F = I + dolfin.grad(U)
            C = F.T * F
        else:
            assert (C.ufl_shape[0] == C.ufl_shape[1])
            dim = C.ufl_shape[0]
            I = dolfin.Identity(dim)

        JF    = dolfin.sqrt(dolfin.det(C))
        IC    = dolfin.tr(C)
        I2    = 1./2. * (IC**2 - dolfin.tr(C.T * C))
        C_inv = dolfin.inv(C)

        Psi   = self.C1 * (IC - dim) + self.C2 * (I2 - dim) - 2 * (self.C1 + 2*self.C2) * dolfin.ln(JF)
        Sigma = 2 * self.C1 * I + 2 * self.C2 * (IC * I - C) - 2 * (self.C1 + 2*self.C2) * C_inv

        return Psi, Sigma
