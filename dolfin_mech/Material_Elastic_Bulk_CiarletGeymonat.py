#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Material_Elastic_Bulk import BulkElasticMaterial

################################################################################

class CiarletGeymonatBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            parameters):

        self.lmbda = self.get_lambda_from_parameters(parameters) # MG20180516: in 2d, plane strain



    def get_free_energy(self,
            U=None,
            C=None):

        C  = self.get_C_from_U_or_C(U, C)
        JF = dolfin.sqrt(dolfin.det(C)) # MG20200207: Watch out! This is well defined for inverted elements!

        Psi   = (self.lmbda/4) * (JF**2 - 1 - 2*dolfin.ln(JF)) # MG20180516: in 2d, plane strain
        Sigma = 2*dolfin.diff(Psi, C)

        # C_inv = dolfin.inv(C)
        # Sigma = (self.lmbda/2) * (JF**2 - 1) * C_inv

        return Psi, Sigma
