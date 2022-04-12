#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
from matplotlib import scale

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class WskelPoroOperator(Operator):

    def __init__(self,
            kinematics,
            Phis0,
            U_test,
            material_parameters,
            material_scaling,
            measure):

        self.kinematics = kinematics
        self.solid_material = dmech.WskelLungElasticMaterial(
            kinematics=kinematics,
            parameters=material_parameters)
        self.material = dmech.PorousElasticMaterial(
            kinematics=kinematics,
            solid_material=self.solid_material,
            scaling=material_scaling,
            Phis0=Phis0)
        self.measure = measure

        dE_test = dolfin.derivative(
            self.kinematics.E, self.kinematics.U, U_test)
        self.res_form = dolfin.inner(self.material.Sigma, dE_test) * self.measure
