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

class MicroHyperElasticityOperator(Operator):

    def __init__(self,
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

        # self.res_form = dolfin.derivative(Psi, U, U_test) * self.measure

        # dE_test = dolfin.derivative(
        #     Psi, U, U_test)
        # self.res_form = dolfin.inner(self.Sigma, dE_test) * self.measure

        self.res_form = dolfin.inner(self.sigma, dolfin.nabla_grad(U_test)) * self.measure

