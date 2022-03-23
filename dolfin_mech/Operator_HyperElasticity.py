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

class HyperElasticityOperator(Operator):

    def __init__(self,
            kinematics,
            U_test,
            material_model,
            material_parameters,
            measure,
            formulation="PK1"): # PK1 or PK2

        self.kinematics = kinematics
        self.material   = dmech.material_factory(kinematics, material_model, material_parameters)
        self.measure    = measure

        assert (formulation in ("PK1", "PK2")),\
            "\"formulation\" should be \"PK1\" or \"PK2\". Aborting."

        if (formulation == "PK2"):
            dE_test = dolfin.derivative(
                kinematics.E, kinematics.U, U_test)
            self.res_form = dolfin.inner(self.material.Sigma, dE_test) * self.measure
        elif (formulation == "PK1"):
            dF_test = dolfin.derivative(
                kinematics.F, kinematics.U, U_test)
            self.res_form = dolfin.inner(self.material.P, dF_test) * self.measure
