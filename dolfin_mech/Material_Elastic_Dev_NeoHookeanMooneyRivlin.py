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
from .Material_Elastic_Dev import DevElasticMaterial

################################################################################

class NeoHookeanMooneyRivlinDevElasticMaterial(DevElasticMaterial):



    def __init__(self,
            parameters):

        if ("mu" in parameters):
            mu = parameters["mu"]
            parameters["C1"] = mu/4
            parameters["C2"] = mu/4
        elif ("E" in parameters) and ("nu" in parameters):
            E  = parameters["E"]
            nu = parameters["nu"]
            mu = E/2/(1+nu)
            parameters["C1"] = mu/4
            parameters["C2"] = mu/4

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
