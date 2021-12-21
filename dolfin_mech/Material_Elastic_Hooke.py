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

class HookeElasticMaterial(ElasticMaterial):



    def __init__(self,
            parameters,
            dim=3,
            PS=False):

        self.lmbda = self.get_lambda_from_parameters(parameters, dim, PS)
        self.mu    = self.get_mu_from_parameters(parameters)



    def get_free_energy(self,
            U=None,
            epsilon=None):

        epsilon = self.get_epsilon_from_U_or_epsilon(
            U, epsilon)

        psi = (self.lmbda/2) * dolfin.tr(epsilon)**2 + self.mu * dolfin.inner(epsilon, epsilon)
        sigma = dolfin.diff(psi, epsilon)

        # assert (epsilon.ufl_shape[0] == epsilon.ufl_shape[1])
        # dim = epsilon.ufl_shape[0]
        # I = dolfin.Identity(dim)
        # sigma = self.lmbda * dolfin.tr(epsilon) * I + 2 * self.mu * epsilon

        return psi, sigma
