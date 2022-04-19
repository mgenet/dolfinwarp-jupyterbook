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

################################################################################

class MacroHyperElasticityOperator(Operator):

    def __init__(self,
            sgma,
            sgma_test,
            U,
            U_test,
            kinematics,
            elastic_behavior,
            measure):

        Psi, self.Sigma = elastic_behavior.get_free_energy(
            C=kinematics.C)
        self.PK1   = kinematics.F * self.Sigma
        self.sigma = self.PK1 * kinematics.F.T / kinematics.J

        self.measure = measure
   
        
        self.res_form = dolfin.derivative(Psi, U,  U_test) * self.measure
                    #   + dolfin.inner(sgma - self.Sigma, sgma_test) * self.measure\

