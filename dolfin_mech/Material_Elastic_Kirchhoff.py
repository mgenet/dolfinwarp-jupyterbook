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

class KirchhoffElasticMaterial(ElasticMaterial):



    def __init__(self,
            parameters):

        self.lmbda = self.get_lambda_from_parameters(parameters)
        self.mu    = self.get_mu_from_parameters(parameters)



    def get_free_energy(self,
            U=None,
            C=None,
            E=None):

        E = self.get_E_from_U_C_or_E(U, C, E)

        Psi = (self.lmbda/2) * dolfin.tr(E)**2 + self.mu * dolfin.inner(E, E)
        Sigma = dolfin.diff(Psi, E)

        # assert (E.ufl_shape[0] == E.ufl_shape[1])
        # dim = E.ufl_shape[0]
        # I = dolfin.Identity(dim)
        # Sigma = self.lmbda * dolfin.tr(E) * I + 2 * self.mu * E

        return Psi, Sigma
