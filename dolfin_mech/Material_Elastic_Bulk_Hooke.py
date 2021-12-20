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

        if ("lambda" in parameters) and ("mu" in parameters):
            lmbda = parameters["lambda"]
            mu    = parameters["mu"]
        elif ("E" in parameters) and ("nu" in parameters):
            E  = parameters["E"]
            nu = parameters["nu"]
            if (dim == 2) and (PS):
                lmbda = E*nu/(1+nu)/(1-  nu)
            else:
                lmbda = E*nu/(1+nu)/(1-2*nu)
            mu = E/2/(1+nu)
        else:
            assert (0),\
                "No parameter found: \"+str(parameters)+\". Need to provide lambda & mu or E & nu. Aborting."
        if (dim == 2):
            self.K = dolfin.Constant((2*lmbda+2*mu)/2)
        elif (dim == 3):
            self.K = dolfin.Constant((3*lmbda+2*mu)/3)
        else:
            assert (0),\
                "Only in 2D & 3D. Aborting."



    def get_free_energy(self,
            U=None,
            epsilon=None,
            epsilon_sph=None):

        if (U is not None) and (epsilon is None) and (epsilon_sph is None):
            dim = U.ufl_shape[0]
            epsilon = dolfin.sym(dolfin.grad(U))
            I = dolfin.Identity(dim)
            epsilon_sph = dolfin.tr(epsilon)/dim * I
        elif (U is None) and (epsilon is not None) and (epsilon_sph is None):
            assert (epsilon.ufl_shape[0] == epsilon.ufl_shape[1])
            dim = epsilon.ufl_shape[0]
            I = dolfin.Identity(dim)
            epsilon_sph = dolfin.tr(epsilon)/dim * I
        elif (U is None) and (epsilon is None) and (epsilon_sph is not None):
            assert (epsilon_sph.ufl_shape[0] == epsilon_sph.ufl_shape[1])
            dim = epsilon_sph.ufl_shape[0]
        else:
            assert (0),\
                "Need to provide U, epsilon or epsilon_sph. Aborting."

        psi = (dim*self.K/2) * dolfin.tr(epsilon_sph)**2
        sigma = dim*self.K * epsilon_sph

        return psi, sigma
