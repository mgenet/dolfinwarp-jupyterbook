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

        if ("C2" in parameters):
            self.C2 = dolfin.Constant(parameters["C2"])
        elif ("mu" in parameters):
            mu = dolfin.Constant(parameters["mu"])
            self.C2 = mu/2
        elif ("E" in parameters) and ("nu" in parameters):
            E  = dolfin.Constant(parameters["E"])
            nu = dolfin.Constant(parameters["nu"])
            mu = E/2/(1+nu)
            self.C2 = mu/2



    def get_free_energy(self,
            U=None,
            C=None):

        assert (U is not None) or (C is not None), "Must provide U or C. Aborting."
        if (U is not None):
            dim = U.ufl_shape[0]
            I = dolfin.Identity(dim)
            F = I + dolfin.grad(U)
            JF = dolfin.det(F)
            C = F.T * F
        elif (C is not None):
            assert (C.ufl_shape[0] == C.ufl_shape[1])
            dim = C.ufl_shape[0]
            I = dolfin.Identity(dim)
            JF = dolfin.sqrt(dolfin.det(C)) # MG20200207: Watch out! This is well defined for inverted elements!

        IC    = dolfin.tr(C)
        IIC   = (dolfin.tr(C)*dolfin.tr(C) - dolfin.tr(C*C))/2
        C_inv = dolfin.inv(C)

        if   (dim == 2):
            Psi   =   self.C2 * (IIC + IC - 3 - 4*dolfin.ln(JF)) # MG20200206: plane strain
            Sigma = 2*self.C2 * (IC * I - C + I - 2*C_inv)       # MG20200206: plane strain
        elif (dim == 3):
            Psi   =   self.C2 * (IIC - 3 - 4*dolfin.ln(JF))
            Sigma = 2*self.C2 * (IC * I - C - 2*C_inv)

        return Psi, Sigma
