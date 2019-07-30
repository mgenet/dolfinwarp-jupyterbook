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
from .Material_Elastic import ElasticMaterial

################################################################################

class CiarletGeymonatNeoHookeanElasticMaterial(ElasticMaterial):



    def __init__(self,
            parameters):

        self.bulk = dcm.CiarletGeymonatBulkElasticMaterial(parameters)
        self.dev = dcm.NeoHookeanDevElasticMaterial(parameters)



    def get_free_energy(self,
            U=None,
            C=None):

        Psi_bulk, Sigma_bulk = self.bulk.get_free_energy(
            U=U,
            C=C)
        Psi_dev, Sigma_dev = self.dev.get_free_energy(
            U=U,
            C=C)

        Psi   = Psi_bulk   + Psi_dev
        Sigma = Sigma_bulk + Sigma_dev

        return Psi, Sigma
