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

class NeoHookeanDevElasticMaterial(DevElasticMaterial):



    def __init__(self,
            parameters):

        if ("mu" in parameters):
            self.mu = dolfin.Constant(parameters["mu"])
        elif ("E" in parameters) and ("nu" in parameters):
            self.E  = dolfin.Constant(parameters["E"])
            self.nu = dolfin.Constant(parameters["nu"])
            self.mu = self.E/2/(1+self.nu)



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
        C_inv = dolfin.inv(C)

        Psi   = (self.mu/2) * (IC - dim - 2*dolfin.ln(JF)) # MG20180516: in 2d, plane strain
        Sigma =  self.mu    * (I - C_inv)

        return Psi, Sigma
