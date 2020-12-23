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

class CiarletGeymonatNeoHookeanElasticMaterial(ElasticMaterial):



    def __init__(self,
            parameters):

        self.bulk = dmech.CiarletGeymonatBulkElasticMaterial(parameters)
        self.dev  = dmech.NeoHookeanDevElasticMaterial(parameters)



    def get_free_energy(self,
            *args,
            **kwargs):

        Psi_bulk, Sigma_bulk = self.bulk.get_free_energy(
            *args,
            **kwargs)
        Psi_dev, Sigma_dev = self.dev.get_free_energy(
            *args,
            **kwargs)

        Psi   = Psi_bulk   + Psi_dev
        Sigma = Sigma_bulk + Sigma_dev

        return Psi, Sigma
