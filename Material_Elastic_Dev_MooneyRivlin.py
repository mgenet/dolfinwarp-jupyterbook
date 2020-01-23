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

        if ("C1" in parameters) and ("C2" in parameters):
            self.C1 = dolfin.Constant(parameters["C1"])
            self.C2 = dolfin.Constant(parameters["C2"])
        elif ("mu" in parameters):
            self.mu = dolfin.Constant(parameters["mu"])
            self.C1 = self.mu/4
            self.C2 = self.mu/4
        elif ("E" in parameters) and ("nu" in parameters):
            self.E  = dolfin.Constant(parameters["E"])
            self.nu = dolfin.Constant(parameters["nu"])
            self.mu = self.E/2/(1+self.nu)
            self.C1 = self.mu/4
            self.C2 = self.mu/4



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
        IC0   = dolfin.tr(I)
        IIC   = (dolfin.tr(C)**2 - dolfin.tr(C*C))/2
        IIC0  = (dolfin.tr(I)**2 - dolfin.tr(I*I))/2
        C_inv = dolfin.inv(C)

        Psi   =   self.C1 * (IC - IC0 - 2*dolfin.ln(JF)) # MG20180516: in 2d, plane strain
        Sigma = 2*self.C1 * (I - C_inv)

        Psi   +=   self.C2 * (IIC - IIC0 - 4*dolfin.ln(JF))
        Sigma += 2*self.C2 * (IC * I - C - 2*C_inv)

        return Psi, Sigma
