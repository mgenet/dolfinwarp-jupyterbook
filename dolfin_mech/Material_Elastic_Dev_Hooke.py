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
            parameters,
            PS=False):

        if ("mu" in parameters):
            mu = parameters["mu"]
        elif ("E" in parameters) and ("nu" in parameters):
            E  = parameters["E"]
            nu = parameters["nu"]
            mu = E/2/(1+nu)
        else:
            assert (0),\
                "No parameter found: \"+str(parameters)+\". Need to provide mu or E & nu. Aborting."
        self.G = dolfin.Constant(mu)



    def get_free_energy(self,
            U=None,
            epsilon=None,
            epsilon_dev=None):

        if (U is not None) and (epsilon is None) and (epsilon_dev is None):
            dim = U.ufl_shape[0]
            epsilon = dolfin.sym(dolfin.grad(U))
            I = dolfin.Identity(dim)
            epsilon_sph = dolfin.tr(epsilon)/dim * I
            epsilon_dev = epsilon - epsilon_sph
        elif (U is None) and (epsilon is not None) and (epsilon_dev is None):
            assert (epsilon.ufl_shape[0] == epsilon.ufl_shape[1])
            dim = epsilon.ufl_shape[0]
            I = dolfin.Identity(dim)
            epsilon_sph = dolfin.tr(epsilon)/dim * I
            epsilon_dev = epsilon - epsilon_sph
        elif (U is None) and (epsilon is None) and (epsilon_dev is not None):
            assert (epsilon_dev.ufl_shape[0] == epsilon_dev.ufl_shape[1])
            dim = epsilon_dev.ufl_shape[0]
        else:
            assert (0),\
                "Need to provide U, epsilon or epsilon_dev. Aborting."

        psi = self.G * dolfin.inner(epsilon_dev, epsilon_dev)
        sigma = 2*self.G * epsilon_dev

        return psi, sigma
