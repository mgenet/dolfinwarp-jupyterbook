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

class KirchhoffElasticMaterial(ElasticMaterial):



    def __init__(self,
            parameters):

        if ("lambda" in parameters) and ("mu" in parameters):
            self.lmbda = dolfin.Constant(parameters["lambda"])
            self.mu    = dolfin.Constant(parameters["mu"])
        elif ("E" in parameters) and ("nu" in parameters):
            self.E  = dolfin.Constant(parameters["E"])
            self.nu = dolfin.Constant(parameters["nu"])
            self.lmbda = self.E*self.nu/(1+self.nu)/(1-2*self.nu) # MG20180516: in 2d, plane strain
            self.mu    = self.E/2/(1+self.nu)



    def get_free_energy(self,
            U=None,
            C=None,
            E=None):

        if (E is None):
            if (C is None):
                dim = U.ufl_shape[0]
                I = dolfin.Identity(dim)
                F = I + dolfin.grad(U)
                C = F.T * F
            else:
                assert (C.ufl_shape[0] == C.ufl_shape[1])
                dim = C.ufl_shape[0]
                I = dolfin.Identity(dim)
            E = (C - I)/2
        else:
            assert (E.ufl_shape[0] == E.ufl_shape[1])
            dim = E.ufl_shape[0]
            I = dolfin.Identity(dim)

        Psi = (self.lmbda/2) * dolfin.tr(E)**2 + self.mu * dolfin.inner(E,E)

        Sigma = self.lmbda * dolfin.tr(E) * I + 2 * self.mu * E

        return Psi, Sigma
