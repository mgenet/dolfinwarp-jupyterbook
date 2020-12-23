#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2020                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019-2020                                              ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin

import dolfin_mech as dmech
from .Material_Elastic_Bulk import BulkElasticMaterial

################################################################################

class PneumoBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            parameters):

        self.alpha = dolfin.Constant(parameters["alpha"])
        self.gamma = dolfin.Constant(parameters["gamma"])



    def get_free_energy(self,
            U=None,
            C=None):

        assert (U is not None) or (C is not None), "Must provide U or C. Aborting."
        if (U is not None):
            dim = U.ufl_shape[0]
            I = dolfin.Identity(dim)
            F = I + dolfin.grad(U)
            JF = dolfin.det(F)
            C = F.T * F
        elif (C is not None):
            JF = dolfin.sqrt(dolfin.det(C)) # MG20200207: Watch out! This is well defined for inverted elements!

        IC    = dolfin.tr(C)
        C_inv = dolfin.inv(C)

        Psi   = (self.alpha) * (dolfin.exp(self.gamma*(JF**2 - 1 - 2*dolfin.ln(JF))) - 1)
        Sigma = (self.alpha) *  dolfin.exp(self.gamma*(JF**2 - 1 - 2*dolfin.ln(JF))) * (2*self.gamma) * (JF**2 - 1) * C_inv

        return Psi, Sigma
