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
from .Material_Elastic_Bulk import BulkElasticMaterial

################################################################################

class CiarletGeymonatBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            parameters):

        if ("lambda" in parameters):
            self.lmbda = dolfin.Constant(parameters["lambda"])
        elif ("E" in parameters) and ("nu" in parameters):
            E  = dolfin.Constant(parameters["E"])
            nu = dolfin.Constant(parameters["nu"])
            self.lmbda = E*nu/(1+nu)/(1-2*nu) # MG20180516: in 2d, plane strain



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
            JF = dolfin.sqrt(dolfin.det(C)) # MG20200207: Watch out! This is well defined for inverted elements!

        IC    = dolfin.tr(C)
        C_inv = dolfin.inv(C)

        Psi   = (self.lmbda/4) * (JF**2 - 1 - 2*dolfin.ln(JF)) # MG20180516: in 2d, plane strain
        Sigma = (self.lmbda/2) * (JF**2 - 1) * C_inv

        return Psi, Sigma
