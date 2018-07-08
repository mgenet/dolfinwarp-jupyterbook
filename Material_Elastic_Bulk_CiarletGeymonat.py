#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018                                            ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_cm as dcm
from Material_Elastic_Bulk import BulkElasticMaterial

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
            C):

        JF    = dolfin.sqrt(dolfin.det(C))
        IC    = dolfin.tr(C)
        C_inv = dolfin.inv(C)

        Psi   = (self.lmbda/4) * (JF**2 - 1 - 2*dolfin.ln(JF)) # MG20180516: in 2d, plane strain
        Sigma = (self.lmbda/2) * (JF**2 - 1) * C_inv

        return Psi, Sigma
