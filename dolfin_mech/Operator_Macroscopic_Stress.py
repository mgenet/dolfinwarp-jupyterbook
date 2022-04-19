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
from .Operator import Operator
from .Problem import Problem

################################################################################

class MacroscopicStress(Operator):

    def __init__(self,
            sigma,
            sigma_test,
            measure,
            kinematics,
            elastic_behavior):

        Psi, self.Sigma = elastic_behavior.get_free_energy(
            C=kinematics.C)
        self.PK1 = kinematics.F * self.Sigma
        sigma_U = self.PK1 * kinematics.F.T / kinematics.J

        self.measure = measure
        self.res_form = dolfin.inner(sigma - sigma_U, sigma_test) * self.measure
        # self.res_form = dolfin.inner(sigma_U - sigma, sigma_test) * self.measure
        # self.res_form = (self.sigma_U[0,0] - sigma[0,0]) * sigma_test[0,0] * self.measure\
        #               + (self.sigma_U[1,1] - sigma[1,1]) * sigma_test[1,1] * self.measure\
        #               + (self.sigma_U[1,0] - sigma[1,0]) * sigma_test[1,0] * self.measure\
        #               + (self.sigma_U[0,1] - sigma[0,1]) * sigma_test[0,1] * self.measure
        
        print("macroscopic stress variable is added")

