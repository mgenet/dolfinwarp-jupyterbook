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

class LinearizedElasticityOperator(Operator):

    def __init__(self,
            u,
            u_test,
            kinematics,
            elastic_behavior,
            measure):

        psi, self.sigma = elastic_behavior.get_free_energy(
                epsilon=kinematics.epsilon)

        epsilon_test = dolfin.derivative(
            kinematics.epsilon,
            u,
            u_test)

        self.measure = measure
        self.res_form = dolfin.inner(
            self.sigma,
            epsilon_test) * self.measure
