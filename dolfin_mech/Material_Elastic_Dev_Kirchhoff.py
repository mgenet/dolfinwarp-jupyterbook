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

class KirchhoffDevElasticMaterial(DevElasticMaterial):



    def __init__(self,
            parameters):

        self.G = self.get_G_from_parameters(parameters)



    def get_free_energy(self,
            U=None,
            C=None,
            E=None,
            E_dev=None):

        E_dev = self.get_E_dev_from_U_C_E_or_E_dev(
            U, C, E, E_dev)
        
        Psi   =   self.G * dolfin.inner(E_dev, E_dev)
        Sigma = 2*self.G *              E_dev

        return Psi, Sigma
