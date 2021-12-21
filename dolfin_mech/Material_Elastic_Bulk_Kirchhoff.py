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

class KirchhoffBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            parameters,
            dim=3,
            PS=False):

        self.K = self.get_K_from_parameters(parameters, dim, PS)



    def get_free_energy(self,
            U=None,
            C=None,
            E=None,
            E_sph=None):

        E_sph = self.get_E_sph_from_U_C_E_or_E_sph(
            U, C, E, E_sph)
        assert (E_sph.ufl_shape[0] == E_sph.ufl_shape[1])
        dim = E_sph.ufl_shape[0]

        Psi   = (dim*self.K/2) * dolfin.tr(E_sph)**2
        Sigma =  dim*self.K    *           E_sph

        return Psi, Sigma
