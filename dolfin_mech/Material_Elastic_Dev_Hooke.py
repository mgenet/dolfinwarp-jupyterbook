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

class HookeDevElasticMaterial(DevElasticMaterial):



    def __init__(self,
            parameters):

        self.G = self.get_G_from_parameters(parameters)



    def get_free_energy(self,
            U=None,
            epsilon=None,
            epsilon_dev=None):

        epsilon_dev = self.get_epsilon_dev_from_U_epsilon_or_epsilon_dev(
            U, epsilon, epsilon_dev)
        
        psi   =   self.G * dolfin.inner(epsilon_dev, epsilon_dev)
        sigma = 2*self.G *              epsilon_dev

        return psi, sigma
