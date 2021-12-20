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
from .Material_Elastic import ElasticMaterial

################################################################################

class CiarletGeymonatNeoHookeanMooneyRivlinElasticMaterial(ElasticMaterial):



    def __init__(self,
            parameters):

        self.bulk = dmech.CiarletGeymonatBulkElasticMaterial(parameters)
        self.dev  = dmech.NeoHookeanMooneyRivlinDevElasticMaterial(parameters)



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
