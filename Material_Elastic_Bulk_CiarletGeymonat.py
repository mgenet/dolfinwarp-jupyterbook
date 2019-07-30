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
from .Material_Elastic_Bulk import BulkElasticMaterial

################################################################################

class CiarletGeymonatBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            parameters):

        if ("lambda" in parameters):
            self.lmbda = dolfin.Constant(parameters["lambda"])
        elif ("E" in parameters) and ("nu" in parameters):
            self.E     = dolfin.Constant(parameters["E"])
            self.nu    = dolfin.Constant(parameters["nu"])
            self.lmbda = self.E*self.nu/(1+self.nu)/(1-2*self.nu) # MG20180516: in 2d, plane strain



    def get_free_energy(self,
            U=None,
            C=None):

        if (C is None):
            dim = U.ufl_shape[0]
            I = dolfin.Identity(dim)
            F = I + dolfin.grad(U)
            C = F.T * F

        JF    = dolfin.sqrt(dolfin.det(C))
        IC    = dolfin.tr(C)
        C_inv = dolfin.inv(C)

        Psi   = (self.lmbda/4) * (JF**2 - 1 - 2*dolfin.ln(JF)) # MG20180516: in 2d, plane strain
        Sigma = (self.lmbda/2) * (JF**2 - 1) * C_inv

        return Psi, Sigma
