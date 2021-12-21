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

class NeoHookeanDevElasticMaterial(DevElasticMaterial):



    def __init__(self,
            parameters):

        self.C1 = self.get_C1_from_parameters(parameters)



    def get_free_energy(self,
            U=None,
            C=None):

        C     = self.get_C_from_U_or_C(U,C)
        IC    = dolfin.tr(C)
        JF    = dolfin.sqrt(dolfin.det(C)) # MG20200207: Watch out! This is well defined for inverted elements!
        # C_inv = dolfin.inv(C)

        assert (C.ufl_shape[0] == C.ufl_shape[1])
        dim = C.ufl_shape[0]
        # I = dolfin.Identity(dim)

        if   (dim == 2):
            Psi   =   self.C1 * (IC - 2 - 2*dolfin.ln(JF)) # MG20200206: plane strain
            # Sigma = 2*self.C1 * (I - C_inv)
        elif (dim == 3):
            Psi   =   self.C1 * (IC - 3 - 2*dolfin.ln(JF))
            # Sigma = 2*self.C1 * (I - C_inv)
        Sigma = 2*dolfin.diff(Psi, C)

        return Psi, Sigma
