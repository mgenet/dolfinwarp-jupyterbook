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
from .Material_Elastic_Dev import DevElasticMaterial

################################################################################

class MooneyRivlinDevElasticMaterial(DevElasticMaterial):



    def __init__(self,
            parameters):

        if ("c1" in parameters):
            self.c1 = dolfin.Constant(parameters["c1"])
        if ("c2" in parameters):
            self.c2 = dolfin.Constant(parameters["c2"])



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

        Psi   = self.c1 * (IC - dim) + self.c2 * (I2 - dim) - 2 * (self.c1 + 2*self.c2) * dolfin.ln(JF)
        Sigma = 2 * self.c1 * I + 2 * self.c2 * (IC * I - C) - 2 * (self.c1 + 2*self.c2) * C_inv

        return Psi, Sigma
