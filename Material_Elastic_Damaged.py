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

import dolfin_cm as dcm
from .Material_Elastic import ElasticMaterial

################################################################################

class DamagedElasticMaterial(ElasticMaterial):



    def __init__(self,
            elastic_material):

        self.elastic_material = elastic_material



    def get_free_energy(self,
            epsilon,
            d):

        psi, sigma = self.elastic_material.get_free_energy(
            epsilon=epsilon)

        psi   *= (1 - d)
        sigma *= (1 - d)

        return psi, sigma
