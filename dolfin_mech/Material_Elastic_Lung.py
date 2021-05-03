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

import dolfin_mech as dmech
from .Material_Elastic import ElasticMaterial

################################################################################

class LungElasticMaterial(ElasticMaterial):



    def __init__(self,
            parameters,
            version):

        if version == 1:
            assert set(["alpha", "gamma", "mu"]).issubset(set(parameters.keys()))
            self.bulk = dmech.PneumoBulkElasticMaterial(parameters)
            self.dev = dmech.NeoHookeanDevElasticMaterial(parameters)
        elif version == 2:
            assert set(["alpha", "gamma", "c1", "c2"]).issubset(set(parameters.keys()))
            self.bulk = dmech.PneumoBulkElasticMaterial(parameters)
            self.dev = dmech.MooneyRivlinDevElasticMaterial(parameters)



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
