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

class CiarletGeymonatNeoHookeanElasticMaterial(ElasticMaterial):



    def __init__(self,
            kinematics,
            parameters,
            decoup=False):

        self.kinematics = kinematics

        self.bulk = dmech.CiarletGeymonatElasticMaterial(kinematics, parameters)
        self.dev  = dmech.NeoHookeanElasticMaterial(kinematics, parameters, decoup)

        self.Psi   = self.bulk.Psi   + self.dev.Psi
        self.Sigma = self.bulk.Sigma + self.dev.Sigma
        self.P     = self.bulk.P     + self.dev.P
        self.sigma = self.bulk.sigma + self.dev.sigma



    # def get_free_energy(self, *args, **kwargs):

    #     Psi_bulk, Sigma_bulk = self.bulk.get_free_energy(*args, **kwargs)
    #     Psi_dev , Sigma_dev  = self.dev.get_free_energy(*args, **kwargs)

    #     Psi   = Psi_bulk   + Psi_dev
    #     Sigma = Sigma_bulk + Sigma_dev

    #     return Psi, Sigma



    # def get_PK2_stress(self, *args, **kwargs):

    #     Sigma_bulk = self.bulk.get_PK2_stress(*args, **kwargs)
    #     Sigma_dev  = self.dev.get_PK2_stress(*args, **kwargs)

    #     Sigma = Sigma_bulk + Sigma_dev

    #     return Sigma



    # def get_PK1_stress(self, *args, **kwargs):

    #     P_bulk = self.bulk.get_PK1_stress(*args, **kwargs)
    #     P_dev  = self.dev.get_PK1_stress(*args, **kwargs)

    #     P = P_bulk + P_dev

    #     return P
