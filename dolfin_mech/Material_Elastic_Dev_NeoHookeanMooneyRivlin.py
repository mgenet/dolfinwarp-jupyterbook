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
from .Material_Elastic_Dev import DevElasticMaterial

################################################################################

class NeoHookeanMooneyRivlinDevElasticMaterial(DevElasticMaterial):



    def __init__(self,
            parameters):

        C1,C2 = self.get_C1_and_C2_from_parameters(parameters)
        parameters["C1"] = C1
        parameters["C2"] = C2

        self.nh = dmech.NeoHookeanDevElasticMaterial(parameters)
        self.mr = dmech.MooneyRivlinDevElasticMaterial(parameters)



    def get_free_energy(self,
            *args,
            **kwargs):

        Psi_nh, Sigma_nh = self.nh.get_free_energy(
            *args,
            **kwargs)
        Psi_mr, Sigma_mr = self.mr.get_free_energy(
            *args,
            **kwargs)

        Psi   = Psi_nh   + Psi_mr
        Sigma = Sigma_nh + Sigma_mr

        return Psi, Sigma
