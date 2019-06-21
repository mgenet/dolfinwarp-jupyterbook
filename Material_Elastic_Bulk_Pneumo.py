#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019                                                   ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

from builtins import *

import dolfin

import dolfin_cm as dcm
from .Material_Elastic_Bulk import BulkElasticMaterial

################################################################################

class PneumoBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            parameters):

        self.alpha = dolfin.Constant(parameters["alpha"])
        self.gamma = dolfin.Constant(parameters["gamma"])



    def get_free_energy(self,
            C):

        JF    = dolfin.sqrt(dolfin.det(C))
        IC    = dolfin.tr(C)
        C_inv = dolfin.inv(C)

        Psi   = (self.alpha) * (dolfin.exp(self.gamma*(JF**2 - 1 - 2*dolfin.ln(JF))) - 1)
        Sigma = (self.alpha) * dolfin.exp(self.gamma*(JF**2 - 1 - 2*dolfin.ln(JF))) * (2*self.gamma) * (JF**2 - 1) * C_inv

        return Psi, Sigma
