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

class WbulkPoroOperator(Operator):

    def __init__(self,
            kinematics,
            Phis0,
            Phis,
            U_test,
            Phis_test,
            material_parameters,
            material_scaling,
            measure):

        self.kinematics = kinematics
        self.solid_material = dmech.WbulkLungElasticMaterial(
            kinematics=kinematics,
            Phis=Phis,
            Phis0=Phis0,
            parameters=material_parameters)
        self.material = dmech.PorousElasticMaterial(
            kinematics=kinematics,
            solid_material=self.solid_material,
            scaling=material_scaling,
            Phis0=Phis0)
        self.measure = measure

        dE_test = dolfin.derivative(
            self.kinematics.E, self.kinematics.U, U_test)
        self.res_form = dolfin.inner(
            self.material.dWbulkdPhis * self.kinematics.J * self.kinematics.C_inv,
            dE_test) * self.measure

        self.res_form += self.material.dWbulkdPhis * Phis_test * self.measure
