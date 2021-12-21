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
from .Material_Elastic_Bulk import BulkElasticMaterial

################################################################################

class HookeBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            parameters,
            dim=3,
            PS=False):

        self.K = self.get_K_from_parameters(parameters, dim, PS)



    def get_free_energy(self,
            U=None,
            epsilon=None,
            epsilon_sph=None):

        epsilon_sph = self.get_epsilon_sph_from_U_epsilon_or_epsilon_sph(
            U, epsilon, epsilon_sph)
        assert (epsilon_sph.ufl_shape[0] == epsilon_sph.ufl_shape[1])
        dim = epsilon_sph.ufl_shape[0]

        psi   = (dim*self.K/2) * dolfin.tr(epsilon_sph)**2
        sigma =  dim*self.K    *           epsilon_sph

        return psi, sigma
