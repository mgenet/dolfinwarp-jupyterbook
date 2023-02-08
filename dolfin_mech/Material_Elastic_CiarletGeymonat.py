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

class CiarletGeymonatElasticMaterial(ElasticMaterial):



    def __init__(self,
            kinematics,
            parameters):

        self.kinematics = kinematics

        self.lmbda = self.get_lambda_from_parameters(parameters)

        self.Psi = (self.lmbda/4) * (self.kinematics.J**2 - 1 - 2*dolfin.ln(self.kinematics.J)) # MG20180516: In 2d, plane strain

        self.Sigma = (self.lmbda/2) * (self.kinematics.J**2 - 1) * self.kinematics.C_inv # MG20200206: Cannot differentiate Psi wrt to C because J is not defined as a function of C

        # self.P = dolfin.diff(self.Psi, self.kinematics.F) # MG20220426: Cannot do that for micromechanics problems
        # self.P = (self.lmbda/2) * (self.kinematics.J**2 - 1) * self.kinematics.F_inv.T
        self.P = self.kinematics.F * self.Sigma

        self.sigma = self.P * self.kinematics.F.T / self.kinematics.J


    # def get_free_energy(self,
    #         U=None,
    #         C=None):

    #     C  = self.get_C_from_U_or_C(U, C)
    #     JF = dolfin.sqrt(dolfin.det(C)) # MG20200207: Watch out! This is well defined for inverted elements!

    #     Psi   = (self.lmbda/4) * (JF**2 - 1 - 2*dolfin.ln(JF)) # MG20180516: in 2d, plane strain
    #     Sigma = 2*dolfin.diff(Psi, C)

    #     # C_inv = dolfin.inv(C)
    #     # Sigma = (self.lmbda/2) * (JF**2 - 1) * C_inv

    #     return Psi, Sigma
