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

        if ("lambda" in parameters) and ("mu" in parameters):
            self.lmbda = dolfin.Constant(parameters["lambda"])
            self.mu    = dolfin.Constant(parameters["mu"])
        elif ("E" in parameters) and ("nu" in parameters):
            E  = parameters["E"]
            nu = parameters["nu"]
            if (dim == 2) and (PS):
                self.lmbda = dolfin.Constant(E*nu/(1+nu)/(1-  nu))
            else:
                self.lmbda = dolfin.Constant(E*nu/(1+nu)/(1-2*nu))
            self.mu = dolfin.Constant(E/2/(1+nu))
        else:
            assert (0),\
                "No parameter found: \"+str(parameters)+\". Need to provide lambda & mu or E & nu. Aborting."



    def get_free_energy(self,
            U=None,
            epsilon=None):

        if (U is not None) and (epsilon is None):
            dim = U.ufl_shape[0]
            epsilon = dolfin.sym(dolfin.grad(U))
        elif (U is None) and (epsilon is not None):
            assert (epsilon.ufl_shape[0] == epsilon.ufl_shape[1])
            dim = epsilon.ufl_shape[0]
        else:
            assert (0),\
                "Need to provide U or epsilon. Aborting."

        # epsilon = dolfin.Variable(epsilon) # MG20211219: This does not work?
        psi = (self.lmbda/2) * dolfin.tr(epsilon)**2 + self.mu * dolfin.inner(epsilon, epsilon)
        # sigma = dolfin.diff(psi, epsilon) # MG20211219: This does not work?

        I = dolfin.Identity(dim)
        sigma = self.lmbda * dolfin.tr(epsilon) * I + 2 * self.mu * epsilon

        return psi, sigma
